# 运行指南

## 环境

- Windows、macOS 或 Linux；
- Python 3.10 以上；
- 推荐使用虚拟环境。

## 安装

```powershell
git clone https://github.com/OWNER/drone-swarm-multi-agent-simulation.git
cd drone-swarm-multi-agent-simulation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

将 `OWNER` 替换为仓库所有者。macOS 或 Linux 使用：

```bash
source .venv/bin/activate
```

## 启动

```powershell
python -m streamlit run app/drone_swarm_workbench.py
```

Windows 也可以双击 `run_app.bat`。

## 推荐的离线演示流程

1. 选择“纯规则模式”；
2. 选择“均衡巡逻演示”；
3. 确认 8 架 UAV、120 步、累计覆盖率和均衡风险策略；
4. 点击开始仿真；
5. 查看专家争辩与候选方案；
6. 查看最终路径和单机建议；
7. 拖动时间轴查看当前覆盖热力；
8. 查看事件、会议纪要和重规划；
9. 下载运行报告。

## 报告目录

运行时尝试写入：

```text
outputs/reports/drone_swarm_simulation_report.md
outputs/reports/drone_swarm_handoff.md
outputs/reports/drone_swarm_agent_trace.md
```

报告默认不提交到 Git。没有目录写入权限时，可使用页面下载按钮。

## 常见问题

### 为什么连接测试很快，模型仿真却等待很久？

连接测试只有一次短 JSON 请求。模型仿真会发送完整共享态势和候选方案，并要求生成多角色意见、交叉质询和推荐；每次重规划会议还可能再次请求。

默认超时是 45 秒，客户端最多重试两次。极端情况下，一场会议可能等待三次超时再回退规则模式。录屏或现场展示建议使用纯规则模式。

### 为什么达到 90% 后仍继续运行？

90% 是参考目标，不是停止条件。只要没有运行到用户设置的最大时间步，仿真继续推进。

### 为什么结果与截图不一样？

结果受场景参数、随机种子、风险事件和动态会议影响。历史截图只代表一次样例运行。
