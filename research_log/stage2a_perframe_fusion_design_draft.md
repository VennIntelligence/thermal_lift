# Stage 2a 设计草案：逐帧证据融合 + 鲁棒 DC（草案，待 owner 与主智能体评审）

> 状态：**DRAFT**，2026-07-06。未 commit，未评审。依据 ACL-046～050 与 changelog 顶部"当前有效结论速览"。
> 目标线：偏移校正 cross-FRC 管线下打赢 TGV×drz（**0.702@30µm / 0.356@24µm / cutoff 23.03µm**）。
> 当前基线：v14/C（v6+E3 主线）收敛于 **0.649@30µm / 0.335@24µm / 25.45µm**（20k 即收敛，50k 零增益，ACL-050）。
> 本稿的取舍分支依赖 v15 消融（unroll4 / m32，进行中）——见 §2.4 的分支表。

---

## 1. 差距诊断回顾（为什么是"榨取效率"，不是"忠实性"）

- v14×TGV 内容一致性 0.871@30µm / 0.849@24µm（ACL-050）：两者重建的带内内容高度相同。神经臂不是在幻觉，
  是**对 drizzle 锚多榨出的真实逐半信息略少**（0.649 vs 0.702）。
- TGV 相对现行 solver 的两个结构优势：
  1. **迭代预算**：TGV 100 次 FISTA 外迭代 ×80 内迭代的凸优化收敛；我们只展开 K=2 步、每步 1 次 DC 梯度
     （`unroll.py:196-207`，eta=softplus⁻¹(0.5) 冻结）。一次梯度步不是投影（`unroll.py:22-23` 自己就写了 caveat）。
  2. **逐帧证据 vs 聚合通道**：prox 的条件输入只有 5 通道聚合统计（E3 主线 `solver_no_drizzle=True` →
     cond_channels=5：mean/median/coverage/variance/highpass 的 ↑2x，`dataset.py:420-423`），
     子像素相位信息在聚合时被抹平——这正是 2026-06-24 决策记录里的病根 #1（"输入塌缩子像素相位"）。
     DC 侧虽然逐帧（y_burst 16 帧 + shifts），但每步只有一次梯度、只挑 `solver_m_frames=16` 帧
     （训练与推理同口径，`real_eval._select_solver_eval_frames`），真实 124 帧/半幅的证据大部分没用上。
- 已排除的解释：训练成熟度（50k 零增益）、配准（centered 出口 + 逐半校正后残余 ~0.05 LR px）、
  对齐质量（精修资产已默认）、DR（关案）。

## 2. 设计空间与取舍

### 候选 A（推荐主线）：逐帧 lift + 置换不变融合（proposal §1 "输入"节的落地）

把 burst 的逐帧证据在**网络内部**提升到 HR 网格并做与帧数无关的融合，prox 条件输入从
"5ch 聚合"升级为"5ch 聚合 ⊕ 融合特征 F"。关键点：**在 `UnrolledSolver.forward` 内部做**，
不动 dataset——y_burst(B,M,h,w) 和 shifts(B,M,2) 本来就传进来了（`unroll.py:157-177`）。

数据流（现行 → 之后）：

```
现行: cond = obs5ch(B,5,H,W)            → prox([x(1), cond5]) = 6ch 输入
之后: lift_i = splat(y_i, s_i)          (B,M,2,H,W)   # ch0=值, ch1=coverage；逐帧
      e_i   = Enc_shared(lift_i ⊕ φ_i)  (B,M,E,H,W)   # E=16；φ_i=相位编码 2ch
      F     = [mean_i e, max_i e, std_i e] (B,3E,H,W)  # 置换不变、M 无关
      cond' = [F(3E), obs5ch(5)]         (B,3E+5,H,W)
      prox([x(1), cond']) = 1+3E+5 = 54ch 输入（E=16）
```

伪代码（新模块 `fusion.py`，挂在 UnrolledSolver 上）：

```python
class PerFrameFusion(nn.Module):
    def __init__(self, feat_channels=16, frame_chunk=8):
        self.enc = nn.Sequential(  # 共享 3 层小编码器，无 norm（extent 纪律）
            nn.Conv2d(4, feat_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(feat_channels, feat_channels, 3, padding=1),
        )
    def forward(self, y_burst, shifts, scale, hr_shape, frame_mask):
        # 流式统计：sum/sumsq/max 随 chunk 累积 → mean/std/max，M 无关且省显存
        for chunk in split(range(M), self.frame_chunk):
            lift = corner_grid_splat(y_burst[:, chunk], shifts[:, chunk], scale, hr_shape)  # (B,c,2,H,W)
            phase = phase_maps(shifts[:, chunk], scale, hr_shape)                            # (B,c,2,H,W) frac(dx),frac(dy) 常数图
            e = self.enc(cat([lift, phase], dim=2).flatten(0,1)).unflatten(0, (B, c))        # (B,c,E,H,W)
            e = e * frame_mask[:, chunk]        # padded 帧置零，且不进统计分母
            accumulate(e)
        return cat([mean, max, std], dim=1)     # (B,3E,H,W)
```

- **lift 的网格约定是红线**：splat 必须落在 solver x 的**角点网格**（`scale*(i+d)+{0..scale-1}`，
  与 `forward_torch.py:19-21` 的 DC 一致），否则融合特征与 x/DC 差半个 HR 像素，等于把 ACL-049 的坑
  搬进网络内部。实现直接复用 `forward_torch` 的 block 采样索引反向 scatter，并写专门测试（§5）。
- φ_i 相位编码：`frac(dx_i), frac(dy_i)` 作为两张常数图（子像素相位是融合要保留的核心信息；
  聚合统计丢的就是它）。
- 融合每步共享（cond' 不随 k 变，与现行 cond 一致），一次计算 K 步复用。
- M 兼容性：训练 M∈[min_burst,16..32]（`dataset._select_burst` 已随机子采样），真实 124/248 帧
  streaming 统计天然支持；coverage 已在 lift ch1 里，统计对 M 的漂移由 frame_mask 归一化 + 训练期
  M 随机化压制。

### 候选 B：更深展开 + 更大 DC 证据（cheap path）

K 2→4（甚至 6）+ m_frames 16→32/64。**这正是 v15 消融在测的东西**——不需要设计，只需要读数。
优点：零新代码；缺点：每步全场 DC 前向/伴随是主要算力项，K=4 步时约 2× 训练时长；
且它不解决"聚合输入抹平相位"的表征问题，天花板可能就是 TGV 本身（同一份证据、更差的优化器）。

### 候选 C：鲁棒 DC（逐帧 IRLS/Huber 权重 + gain/offset 泄压，proposal "求解器中间步"节）

公式化（非黑箱）逐帧权重：`w_i = huber'(r_i)/r_i`，r_i 为该帧带限 DC 残差；可选逐帧标量 gain/offset
最小二乘泄压（对齐真实 drift 模型）。对接事实：精修后残余 shift 误差 ~0.05 LR px + 真实帧间 drift。
预期是**稳健性**收益（坏帧不再拖累），对干净 248 帧的 FRC 榨取增益可能有限——放在 A 之后做增量。

### 2.4 v15 消融分支表（结果落地后按此定序）

| v15 结果 | 解读 | Stage 2a 执行序 |
|---|---|---|
| unroll4 与 m32 均 ~平（@30µm Δ<0.02） | 迭代/证据预算不是瓶颈 → 表征问题坐实 | **直接上候选 A**（K=2, m=16 不变，单变量） |
| unroll4 有效（Δ≥0.02） | 迭代预算是瓶颈之一 | 先吃 B（K=4 定为新基线），A 在 K=4 上叠加 |
| m32 有效 | DC 证据预算是瓶颈之一 | m 提到 32（训练+推理同口径），A 叠加；A 的融合本来就吃全帧，预期更大 |
| 两者都有效 | 预算全面不足 | B 全吃（K=4+m32）为新基线，再叠 A |

## 3. 纪律红线（全部继承，不翻案）

1. **prox 只填 forward 零空间**：end-on-DC 顺序与冻结 eta 不动（ACL-025/026，`unroll.py:14-23`）。
   融合特征只进 prox 的条件输入，不碰 DC 步。
2. **不做 loss 侧 forward 锚**（ACL-017/019 证伪，`losses.py` forward_model_weight=0 保持）。
3. **保真悬崖警惕**：融合特征让 prox 更强 → 幻觉风险回升。探测器=偏移校正 cross-FRC 的
   self-vs-cross 差值（self 高 cross 不高 = 幻觉复发信号）+ 0d extent 探针（唯一可信探针，ACL-047）。
4. **判据**：主门 = 偏移校正 cross-FRC vs drizzle（24–30µm 段 + 1/7 cutoff；20µm 孔径零点不采信），
   目标 TGV 0.702@30µm / 23.03µm；synth PSNR 仅 sanity 下限（对 C 的 31.26 不塌 >1dB）；
   dc_resid 仍非判据。
5. **band-gated loss（roadmap Step 3）织入方式**：loss 残差在 25–40µm 带内加权（Fourier 环形权重
   或 DoG 带通，σ 由 25.45µm 权威频带换算 HR px），作为**独立单变量实验**（§5 E3）排在融合之后，
   不与 A 同轮引入。
6. 一次一个变量；截断必 log；20µm 契约硬断言不动。

## 4. 显存/算力预算（5090 32GB，batch4 p384，v6 池，20k 步收敛已证）

基线事实：K=2、m=16、batch4 p384 的 20k ≈ 1h（C/v14 实测口径）。

| 候选 | 显存增量估算 | 步时增量 | 20k 训练时长 |
|---|---|---|---|
| A（E=16, chunk=8） | lift+enc 激活 ≈ B4×chunk8×(4+3×16)ch×384² ×4B ≈ 1.2GB/层级，checkpoint 后 <4GB | +30–50%（enc 便宜，DC 仍主导） | ~1.5h |
| B（K=4） | DC create_graph 激活 ×2 ≈ +6–8GB（若 OOM 降 batch 2 并记录） | ~+100% | ~2h |
| B（m=32） | y_burst/plan ×2 ≈ +2–3GB | ~+60%（DC 帧数线性） | ~1.6h |
| C（IRLS 权重） | 忽略 | +5% | ~1h |
| A+B 全叠 | 需 checkpointing，batch 可能降 2 | — | ~3–4h |

真实推理（248 帧全幅 960×1280）：A 的 streaming chunk=8 → 峰值 <6GB，full_halo96 口径不变。

## 5. 实施计划（按"最便宜可证伪"排序）

**E0（零训练，~20 min GPU）**：v14_50k 推理期证据消融——real_eval 推理时把 m_frames 16→64→124
（只改推理选帧，训练不动；`_select_solver_eval_frames` 的 m 参数来自 config，需加 override 入口，~10 行）。
若 cross-FRC @30µm 提升 ≥0.02：证据预算在推理端就有肉，m32 训练臂优先级升。
go/no-go：任何提升都值得记录；无提升也排除一个便宜解释。

**E1（候选 A 主实验，~1.5h/臂）**：PerFrameFusion（E=16, mean⊕max⊕std, chunk=8），
K/m 按 §2.4 分支表定，其余全同 v14。20k 步。
go：@30µm ≥ 0.67（对 C +0.02）且 cutoff ≤25.45 不退、0d extent 探针过、synth ≥30.3。
no-go：平或 self-vs-cross 幻觉信号 → 回看 E 通道数/池化（一次微调机会），再平则候选 A 关闭，转 C+2c。

**E2（候选 C 叠加，~1h）**：E1 最优臂 + 公式化 IRLS/Huber 逐帧权重（`--solver-dc-irls huber:delta`）。
go：@30µm 不降且真实视觉/OOB 改善；主要看真实 248 帧全幅（drift 帧存在）。

**E3（band-gated loss，~1h）**：25–40µm 带权 loss（`--loss-band-um 25,40 --loss-band-weight w`）。
go：cross-FRC 带内提升且 synth sanity 不破。

**文件级改动清单**（E1 为主）：
- 新 `algos/ep07_unet_sr/src/unet_sr/fusion.py`：`corner_grid_splat`（复用 forward_torch 索引反向 scatter）、
  `phase_maps`、`PerFrameFusion`（streaming 统计 + frame_chunk）。
- `unroll.py`：`UnrolledSolver.__init__` 加 `fusion: str = "none"` / `fusion_channels` / `fusion_frame_chunk`；
  `forward` 在循环前算 `F` 并 `cond = cat([F, cond])`；cond_channels 校验改为动态。
- `config.py`：`--solver-fusion {none,perframe}`、`--solver-fusion-channels`、`--solver-fusion-frame-chunk`、
  （E0 用）`--eval-m-frames-override`；校验 + wire。
- `solver_train.py`：透传 + banner 打印 fusion 配置。
- `real_eval.py`：`infer_solver_from_burst_full_halo` 无需改（fusion 在 solver 内部）；E0 的 m 覆写入口。
- 测试（`tests/test_fusion.py`）：①置换不变（帧序 shuffle → 输出逐位一致）；②M 无关（16 帧 vs 32 帧 shape/统计路径）；
  ③**网格约定**（对线性 ramp，corner_grid_splat 的重心 = `scale*(i+d)+0.5`，与 forward block 中心一致——ACL-049 教训的单测化）；
  ④frame_mask 零帧不进统计；⑤fusion=none 与现行逐位等价（回归）。

## 6. 风险清单与探测器

| 风险 | 探测器 |
|---|---|
| 融合特征强化先验 → 保真悬崖复发 | self-FRC 与 cross-FRC 差值监控（self↑cross↓=幻觉）；0d extent 探针；30k+ 步的 cross-FRC 曲线不回落 |
| lift 网格约定偏移（ACL-049 重演） | 测试 ③ 硬断言 + 训后 probe_pair_offset 残余 ~0.05px |
| M 分布漂移（训练 ≤32 vs 真实 124/248） | 推理时对 M∈{16,64,124} 各出一版 cross-FRC，差异 >0.02 即需 M 增广 |
| K=4 OOM / batch 降级破坏可比性 | 若降 batch 必须同时跑 batch2 的 K=2 对照臂，或用梯度累积保等效 batch |
| streaming std 数值误差（大 M） | Welford 累积；单测对 naive 实现逐位 ±1e-5 |
| 池统计被坏帧污染（真实 drift 帧） | frame_mask 对接 0d/坏帧名单；E2 的 IRLS 权重同源复用 |
| 20k 收敛假设在新架构失效 | 训练曲线 + 30k 延长探针（一次性，若 20k/30k 差 >0.01@30µm 则调 schedule） |

## 7. 与 proposal §2 原计划的差异说明

- 原 2a 写"取代固定 phase-bin 通道路线"——E3 主线（solver_no_drizzle）本来就没有 phase-bin 通道，
  故本稿为"5ch 聚合 ⊕ F"而非替换；phase 信息由 φ_i 编码进 lift。
- 原 1a 的 DR 前置已被 ACL-049/050 关案，从依赖链中移除。
- 原判据"real split-half FRC"已换轨为"偏移校正 cross-FRC"（ACL-047/049）。
