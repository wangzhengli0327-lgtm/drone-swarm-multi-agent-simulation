# 抽象复杂环境下无人机集群多智能体协同管控仿真

一个基于 Streamlit 的多智能体教学仿真原型，用于展示无人机集群在抽象网格环境中的共享态势、专家会商、候选方案比较、路径生成、覆盖评估、事件触发和动态重规划。

## 安全边界

本项目只用于抽象仿真、课程研究和教学展示：

- 不连接真实无人机或其他物理设备；
- 不使用真实地理坐标；
- 不生成真实飞控指令；
- 不提供武器使用、目标选择、突防规避或真实部署建议。

## 主要能力

- 60×60 抽象网格场景；
- 多任务区与高风险区配置；
- 多个中文专家角色读取共享态势；
- “提出意见—相互质询—修改候选—安全审查—形成决议”会议流程；
- 多候选方案综合评分；
- 每架 UAV 的抽象任务、建议位置和完整轨迹；
- 累计覆盖率、滚动覆盖率与时间轴热力图；
- 失效、低电量、覆盖下降和风险状态变化触发的动态重规划；
- 本地规则模式与 OpenAI 兼容模型接口实验；
- 会议纪要、事件记录和 Markdown 报告。

## 快速开始

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 启动应用

```powershell
python -m streamlit run app/drone_swarm_workbench.py
```

也可以在 Windows 中双击 `run_app.bat`。

### 3. 离线演示

选择“纯规则模式”即可离线完成场景配置、专家会商、候选比较、仿真、重规划和报告输出。

## 模型接口

应用支持 OpenAI 兼容的 `/chat/completions` 接口。请在界面中配置提供商、Base URL、模型名称和 API Key。

不要把真实密钥写入源码、README、截图、输出报告或 Git 历史。可以复制 `.env.example` 为 `.env`，但 `.env` 已被 `.gitignore` 排除。

“测试模型连接”只发送一次很短的 JSON 请求。正式模型模式会在初始会议及后续重规划会议中调用模型，提示词和返回结构都更大，因此耗时可能明显高于连接测试。现场演示建议优先使用纯规则模式。

## 项目结构

```text
app/
  drone_swarm_workbench.py   Streamlit 入口
  drone_sim/                 会议引擎、场景、随机数和兼容 API 客户端
docs/
  architecture.md            多智能体设计说明
  run-guide.md               运行指南
  api-integration.md         模型接口说明
  agent-prompts.md           角色提示词说明
outputs/reports/             运行时生成报告，不提交真实结果
tests/                       基础测试
```

## 推荐演示预设

“均衡巡逻演示”使用 8 架 UAV、3 个任务区、120 个时间步、累计覆盖率主指标和一个高风险区。目标覆盖率是参考线，不是提前停止条件。

## 已知边界

- 当前规划使用启发式候选生成和综合评分，不保证数学全局最优；
- 模型模式每场会议进行一次结构化调用，不是多个独立大模型进程长期自治运行；
- 仿真结果受场景参数、随机种子、资源数量和风险事件影响；
- 项目属于原型级机制验证，不代表真实系统部署能力。

## 文档

- [架构说明](docs/architecture.md)
- [运行指南](docs/run-guide.md)
- [API 接入说明](docs/api-integration.md)
- [专家角色提示词](docs/agent-prompts.md)

## 开源说明

仓库尚未附加开源许可证。在许可证确定前，默认保留全部权利。
