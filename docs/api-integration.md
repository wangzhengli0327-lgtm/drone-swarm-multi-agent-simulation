# OpenAI 兼容模型接口

## 支持方式

页面侧栏可以临时填写：

- API Key；
- Base URL；
- 模型名称；
- 是否需要 API Key；
- 请求超时。

内置选项包括 Agnes AI、OpenRouter、本地 Ollama 和自定义 OpenAI 兼容接口。DeepSeek 等兼容服务可以通过自定义接口配置。

## Agnes 示例

```text
Base URL: https://apihub.agnes-ai.com/v1
Model: agnes-2.0-flash
```

也可以使用环境变量：

```powershell
$env:AGNES_API_KEY = "replace_with_your_key"
$env:AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
$env:AGNES_MODEL = "agnes-2.0-flash"
```

## 三种会议模式

- 纯规则模式：不调用外部模型；
- 大模型辅助评审：模型生成意见、质询和推荐，规则评分最终选案；
- 大模型决策实验：模型推荐通过硬约束后可成为执行方案，否则回退规则选案。

## 调用次数与等待时间

“测试模型连接”只发送一条短提示词。

正式仿真中，每场初始会议或重规划会议都可能调用一次模型，输入包含共享态势和候选方案，输出要求包含 6 条角色意见、若干评审和候选推荐。

客户端默认：

- 超时 45 秒；
- 最多重试 2 次；
- 对 429、部分 5xx、TLS、连接重置和超时进行退避重试；
- 失败后回退规则会议。

因此，连接测试成功不代表复杂请求会同样快。慢响应、限流、无效 JSON 或多次重规划都可能显著增加总耗时。

## 密钥安全

- 不要把真实密钥写入仓库；
- 不要提交 `.env` 或 Streamlit secrets；
- 不要在截图和录屏中显示密钥；
- 切换提供商时重新填写密钥；
- 已经暴露的密钥应在平台上撤销并重新生成。
