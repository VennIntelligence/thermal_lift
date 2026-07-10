# OOD content 轴 4 新 motif 族——实现偏离记录（2026-07-10，worktree 实现）

> 对应计划：`research_log/ood_content_motif_families_plan.md`。实现全部落在
> `tcforge/src/tcforge/geometry.py`（4 个 bbox 局部光栅化 helper + 4 个 elif 分支 +
> CPU_SCENE_FAMILIES 注册 + motif_weights 未知键校验），测试 3 个新用例进
> `tcforge/tests/test_geometry.py`（116 全绿；golden 字节一致性测试通过 = 现有池字节级不变已验证）。
> 预览：`tmp/make_motif_previews.py` → `tmp/motif_previews/*.png`（每族 2 景，960×1280 HR、
> SSAA 4、inscribe_disc，rot 0 + rot 137）。

## 与计划 §2 规格的偏离（全部预览目测后定）

1. **密度上调**（首版按计划参数 occupancy 仅 0.3-0.5%，48 景池带限 FRC 会噪声主导）：
   blob 数 3-9 → 8-22、基半径 [90,260] → [110,420]µm；text 行数 1-3 → 2-5、
   字形高 [140,260] → [180,400]µm、笔画上限 1.3→1.6×FLOOR。调后 occupancy：
   blobs ~2-3%、text ~1-2%、rings ~8-13%、voronoi ~38-42%。
2. **共享 CPU clutter（passives/vias/edge-IO）对 4 个新族关闭**（计划未覆盖的设计点）：
   混入分布内零件内容会稀释 content 轴要测的语法距离。门控只查 family 名、零 RNG，
   原 6 族 RNG 流不变（golden 测试钉死）。
3. 谐波阶从 2 起（1 阶近似平移，不产生形变）；幅度归一到 amp_total 并按
   amp_total ≤ 1 − 1.5*FLOOR/r0 截断（最窄颈守卫）。
4. 椭圆环的 floor 按**短轴度量宽度**执行（flo = FLOOR/ratio；环宽与间隙均 ≥ flo），
   比计划字面的 "w ≥ FLOOR" 更严。
5. 第二环系拒绝采样放置（两包围圆圆心距 ≥ r1+r2+2*FLOOR，8 次尝试失败则放弃），
   避免双环系交叠产生亚 floor 莫尔楔形。
6. 7 段字形比例硬约束：h ≥ 3t+2*FLOOR、w ≥ 2t+FLOOR（保证平行段净空 ≥ FLOOR）；
   每字符至少 2 段非空（不足强制 A+D）；字符 pitch ≥ 字宽+FLOOR；刻字模式外沿 ≥ 1.5*FLOOR。
7. Voronoi 边界带 (d2−d1) ≤ channel_w 的垂直宽度在种子中点处恰为 channel_w、
   远离中点只会变宽 → 通道处处 ≥ FLOOR；种子间距 ≥ 5*FLOOR 保证胞元内切半径 ≥ 2*FLOOR。
