# Local/Remote Collaboration Protocol

本协议约定本地 Codex 与远端 Windows/WSL 环境的分工方式，目标是减少大文件同步、避免环境漂移，同时保留可复现的代码与实验记录。

## Environments

- Local workspace: `/Users/ujs/mycode/thermal_lift`
- Remote SSH entry: `Administrator@100.98.99.29`
- Remote WSL user: `ujs`
- Remote project path: `~/thermal_lift` (`/home/ujs/thermal_lift` in WSL; moved off the slow `/mnt/c` Windows mount for I/O speed)
- Analysis host drop entry: `ujs@100.118.98.78`
- Analysis host project path: `/Users/ujs/mycode/thermal_lift`
- Code remote: `git@github-venn:VennIntelligence/thermal_lift.git`

## Remote Connection (WSL-over-SSH) — the working recipe

远端 `Administrator@100.98.99.29`(Windows + 5090)的 OpenSSH 默认 shell 是 WSL,且被以 `wsl.exe -c "<cmd>"` 调用,而 `-c` 不是 wsl 的合法参数 —— 所以**非交互式** `exec_command` / `ssh host "<cmd>"` 会直接报"无效的命令行参数,请使用 wsl.exe --help"(且输出是 UTF-16/GBK 乱码)。**别在非交互式调用上耗时间。**

正确方式:**用交互式 shell**(`ssh -tt` 或 paramiko `invoke_shell()`)—— 它直接落进 WSL **bash**(用户 `ujs`,项目 `~/thermal_lift`,Linux WSL2)。要点:
- 发**纯 Linux 命令**(不是 `wsl` / `chcp` / `ver`;在 bash 里它们不存在)。
- 输出按 **UTF-8** 解码。坑:ASCII 文本若误用 `utf-16-le` 解码,会变成"看似合法"的中日韩字符、不报 `�`,从而骗过"自动挑编码"的启发式 —— 固定用 utf-8。
- 用唯一 sentinel(`echo __DONE__`)界定命令输出边界,并正则剥掉 ANSI / bracketed-paste 转义:`\x1b\[…[a-zA-Z]` 和 `\x1b]…;\x07`。

paramiko 最小骨架(**密码不入库** —— 运行时从环境变量/交互读取;长期换公钥,见本协议末段):
```python
import paramiko, time, re
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("100.98.99.29", username="Administrator", password=PW, timeout=12)
ch = c.invoke_shell(width=220, height=60); time.sleep(1.5)
while ch.recv_ready(): ch.recv(65536)               # flush login banner
ansi = re.compile(r'\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][0-9;]*;?[^\x07]*\x07')
def run(cmd, wait=15):
    ch.send(cmd + "\n"); buf=b""; t=time.time()
    while time.time()-t < wait:
        buf += ch.recv(65536) if ch.recv_ready() else b""; time.sleep(0.2)
    return ansi.sub('', buf.decode("utf-8", "replace")).replace('\r','')
print(run("cd ~/thermal_lift && git pull --ff-only && nvidia-smi -L"))
```

**根因修复(一劳永逸)**:把远端 sshd 的 `DefaultShell` 设为 cmd/powershell(或给 wsl 配正确命令选项),之后非交互式 `exec_command` / `rsync` / `sftp` 才能直接用,这条 recipe 就不再需要。

## Division Of Labor

本地负责代码编写、方案思考、轻量数据计算和短时间分析。适合在本地完成的工作包括：

- 阅读代码、搜索上下文、设计改动方案。
- 修改源码、配置、文档、测试。
- 运行不依赖大数据和 GPU 的轻量脚本。
- 创建 Git commit，并通过 GitHub 同步代码。
- 整理远端运行结果，写入文档或研究日志。

远端负责重环境、重数据和长时间运行。适合在远端完成的工作包括：

- 读取 `data/`、`output/` 或其他大体量产物。
- 运行 GPU、CUDA、PyTorch 或长耗时训练/推理任务。
- 运行依赖远端专用 venv、conda 或 Windows/WSL 环境的脚本。
- 保存大型中间结果、模型 checkpoint、图像批量输出和实验缓存。

## Short Remote Tasks

简单、时间较短、目标单一的远端任务，由本地一次性写好 Python 或 shell 脚本，然后发到远端执行。

推荐流程：

1. 在本地创建临时脚本，例如 `tmp/remote_probe.py` 或 `tmp/run_small_check.sh`。
2. 将脚本同步到远端项目的 `tmp/` 下。
3. 在远端执行脚本，输出日志、CSV、JSON 或少量图片。
4. 只取回轻量结果，不取回大数据目录或 checkpoint。
5. 若脚本有长期价值，再移动到 `scripts/`、`algos/*/scripts/` 或文档化；否则保留在 `tmp/`。

短任务应尽量做到幂等，输出目录应显式写入 `tmp/`、`output/` 或任务专属路径，避免覆盖已有实验结果。

## Long Remote Tasks

复杂、长时间、需要远端智能体持续判断的任务，不由本地直接遥控执行。由本地一次性写好 handoff prompt，用户手动贴给远端智能体执行。

长任务应在 `tmux` 的 `0` session 中留窗运行，而不是直接用 one-shot 窗口命令。项目已实测：在默认 `remain-on-exit off` 下，`tmux new-window ... "command | tee log"` 会在命令结束后自动关闭 pane，导致用户无法回看屏幕输出。推荐模板：

```bash
tmux new-window -t 0: -n task_name 'bash -lc '"'"'
set -o pipefail
cd /home/ujs/thermal_lift
command 2>&1 | tee output/task_name/stdout.log
rc=${PIPESTATUS[0]}
echo
echo "[tmux done] exit_code=${rc} log=output/task_name/stdout.log"
exec bash -l
'"'"''
```

执行中可用 `tmux attach -t 0` 查看窗口，或用 `tmux capture-pane -pt 0:task_name -S -200` 快速取最近输出。任务结束后窗口会停在 shell，方便人工检查；检查完再手动 `exit` 或 `tmux kill-window -t 0:task_name`。

推荐流程：

1. 本地在 **git 跟踪的目录**写清任务提示词(如 `docs/REMOTE_ORDERS.md`)。**不要放 `tmp/`** —— 它被 `.gitignore` 忽略,无法随仓库同步到远端;`tmp/` 仅用于不需要同步的临时脚本/探针。
2. 提示词包含目标、当前上下文、必须读取的文件、运行命令、预期产物、验收标准和回传摘要格式。
3. 用户将该提示词贴到远端智能体。
4. 远端智能体在 `~/thermal_lift` 中执行任务。
5. 远端完成后，用户唤醒本地 Codex。
6. 本地 Codex读取远端摘要或拉取轻量结果，继续做代码整理、复盘、提交或下一步规划。

长任务提示词应避免要求远端修改无关文件。涉及算法、网络、loss、训练策略或数据管线改动时，必须同步更新 `research_log/algorithm_changelog.md`。

## Data Transfer

代码同步优先使用 GitHub：

- 本地改代码后 commit/push。
- 远端运行前 pull。
- 远端如果产生需要保留的代码改动，也应通过 Git commit/push 回传。

轻量结果可以使用 `rsync`、`scp`、GitHub 或手动复制：

- 适合回传：日志、CSV、JSON、Markdown、少量 PNG/PDF。
- 不适合回传：`data/`、大型 `output/`、`.venv/`、checkpoint、缓存目录。

当前远端 SSH 默认 shell 可能影响非交互式 `rsync`/`sftp`。在修复前，优先使用 GitHub 或交互式 SSH 执行明确命令。

### Local-to-Analysis-Host Drops

本地 Codex 可以主动向分析主机投放小型实验包，用于让远端/分析环境快速查看最近实验状态。该动作是**投放(drop)**，不是同步(sync)。

允许投放的内容：

- 少量 checkpoint 渲染图，例如每个 run 选 5 个阶段的 center-zoom PNG。
- 对应的小型数值数组，例如压缩 `.npz` 温度图、CSV/JSON manifest、关键指标摘要。
- 简短 README，说明来源 commit、checkpoint 列表、渲染参数和使用限制。

禁止投放或同步的内容：

- 不投放完整 `data/`、完整 `output/`、完整 `outputs/`。
- 不投放训练 checkpoint、`.pt/.pth` 权重、venv 或缓存目录。
- 不使用 `rsync --delete`、目录镜像或任何会删除/覆盖远端 `output/` 的同步命令。

推荐远端落点：

```text
/Users/ujs/mycode/thermal_lift/remote_inbox/<YYYYMMDD>_<topic>/
```

示例：

```bash
ssh ujs@100.118.98.78 'mkdir -p /Users/ujs/mycode/thermal_lift/remote_inbox/20260627_checkpoint_evolution'
scp -r output/remote_drop/20260627_checkpoint_evolution/* \
  ujs@100.118.98.78:/Users/ujs/mycode/thermal_lift/remote_inbox/20260627_checkpoint_evolution/
```

SSH 免输密码应通过公钥完成：本地生成/复用 SSH key，并用 `ssh-copy-id` 安装公钥到分析主机。不要把密码写入仓库、脚本、README、shell alias 或长期保存的明文配置。

## Git Rules

- 本地 remote 使用 `git@github-venn:VennIntelligence/thermal_lift.git`。
- 远端也应使用同一 GitHub SSH host 或等价 deploy key。
- 本地是默认提交源；远端是默认执行源。
- 远端仓库位于 Windows 挂载盘，可能出现换行、权限或索引刷新导致的大量 `M` 状态。提交前必须确认 diff 是真实业务改动。
- 不提交 `.venv/`、`data/`、大型 `output/`、checkpoint 或临时缓存。

## Local Venv

本地允许保留根目录 `.venv`，用于轻量开发、代码检查、notebook 工具和小脚本运行。

本地默认不安装 GPU/CUDA/NVIDIA 驱动类依赖。需要 GPU 或重型算法环境时，转到远端对应 `algos/*/` 环境运行。

## Result Handoff

远端任务完成后，应至少回传以下信息：

- 执行时间和机器环境。
- 使用的 commit SHA 或 diff 摘要。
- 运行命令。
- 输出路径。
- 关键指标或结论。
- 失败、跳过或仍需人工判断的事项。

本地 Codex 接手后，应先确认这些信息，再决定是否拉取结果、更新文档、提交代码或生成下一份 `tmp/*.md` 远端提示词。
