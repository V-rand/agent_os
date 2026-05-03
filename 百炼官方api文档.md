# 百炼 API 接入说明（模型调用与工具调用）

## 1. 文档目的

本文档面向工程实现，说明如何基于阿里云百炼（Model Studio / DashScope）的 OpenAI 兼容接口实现：

1. 普通模型调用
2. 带思考模式的模型调用
3. 基于 Function Calling 的工具调用
4. 并行工具调用
5. 强制/禁止工具调用
6. 面向 Agent Runtime 的推荐接入方式

本文默认采用 **OpenAI 兼容接口**，并明确约束：

- 工具由业务侧自行实现与执行
- 不使用百炼内置联网/内置工具能力来替代本地工具编排
- 多步 ReAct 循环由应用端自己实现，不假设云侧替我们完成整个 Agent Loop

---

## 2. 接入方式总览

百炼支持 OpenAI 兼容调用方式。对于工程实现，建议优先采用这一方式，以便复用现有 OpenAI SDK、消息结构和 Function Calling 流程。

### 2.1 OpenAI 兼容 BASE_URL

按地域选择：

```text
北京:      https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 2.2 HTTP Chat Completions Endpoint

```text
POST /chat/completions
```

完整示例：

```text
POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
```

### 2.3 鉴权

通过百炼 API Key 进行鉴权。工程上建议从环境变量读取，而不是硬编码在代码中。

---

## 3. 模型调用

## 3.1 最小模型调用（Python / OpenAI SDK）

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

resp = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[
        {"role": "system", "content": "你是一个严谨的研究助手。"},
        {"role": "user", "content": "请用一句话介绍百炼 OpenAI 兼容接口。"}
    ]
)

print(resp.choices[0].message.content)
```

## 3.2 HTTP 最小调用示例

```bash
curl -X POST 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $DASHSCOPE_API_KEY' \
  -d '{
    "model": "qwen3.6-plus",
    "messages": [
      {"role": "system", "content": "你是一个严谨的研究助手。"},
      {"role": "user", "content": "请用一句话介绍百炼 OpenAI 兼容接口。"}
    ]
  }'
```

## 3.3 思考模式（enable_thinking）

对于支持混合思考模式的模型，可以通过 `enable_thinking` 控制是否开启思考。需要注意：

- 这不是 OpenAI 标准参数
- **Python OpenAI SDK** 里应通过 `extra_body` 传入
- **Node.js OpenAI SDK** 可作为顶层参数传入
- 某些模型默认开启思考，某些默认关闭
- 流式模式下可通过 `reasoning_content` 与 `content` 分别接收思考内容与回答内容

### Python 示例

```python
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[{"role": "user", "content": "请解释一下什么是 Function Calling"}],
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in completion:
    if not chunk.choices:
        print(chunk.usage)
        continue
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print(delta.reasoning_content, end="")
    if hasattr(delta, "content") and delta.content:
        print(delta.content, end="")
```

### 工程建议

如果系统需要完整保留模型显式思考输出用于日志和调试，应在流式消费时分别记录：

- `reasoning_content`
- `content`
- `usage`

但不要把“是否有 reasoning_content”与“系统是否实现了 ReAct agent loop”混为一谈。`enable_thinking` 解决的是模型回复前是否显式思考，不会自动替你完成多步工具编排。

---

## 4. 工具调用（Function Calling）

## 4.1 基本原理

Function Calling 的流程是两段式或多段式的：

1. 应用向模型发送用户问题和工具列表
2. 模型返回 `tool_calls`，指出要调用哪个工具以及入参
3. 应用端执行工具
4. 应用把工具结果以 `role=tool` 的消息追加回 `messages`
5. 再次调用模型，让模型基于工具结果生成自然语言回复

如果问题需要多轮串行工具调用，则在应用端继续循环。

**重点：工具是由应用端执行的，不是百炼替你执行。**

---

## 4.2 tools 参数格式

工具定义遵循 OpenAI 风格。一个典型工具如下：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

工具设计建议：

- `name` 使用稳定且易读的英文名
- `description` 直接说明什么时候该用它
- `parameters` 尽量收窄，避免模型生成模糊入参
- 不要把过多业务规则塞到 description 里，复杂规则应由应用端校验

---

## 4.3 最小 Function Calling 示例（Python）

```python
from openai import OpenAI
import os
import json

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def get_current_weather(arguments):
    location = arguments["location"]
    return f"{location}今天是晴天。"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "北京天气怎么样？"}
]

resp = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,
)

assistant_msg = resp.choices[0].message
messages.append(assistant_msg)

if assistant_msg.tool_calls:
    tool_call = assistant_msg.tool_calls[0]
    tool_name = tool_call.function.name
    tool_args = json.loads(tool_call.function.arguments)

    if tool_name == "get_current_weather":
        tool_result = get_current_weather(tool_args)
    else:
        raise ValueError(f"unknown tool: {tool_name}")

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": tool_result,
    })

    final_resp = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        tools=tools,
    )

    print(final_resp.choices[0].message.content)
else:
    print(assistant_msg.content)
```

---

## 4.4 串行工具调用循环

当问题存在依赖链时，需要在应用端使用 while 循环驱动串行工具调用。例如：

- 先计算中间结果，再查数据库
- 先查询实体 ID，再用该 ID 调第二个接口
- 先读搜索结果，再决定是否抓正文

推荐逻辑：

```python
while True:
    resp = model(messages, tools)
    assistant_msg = resp.choices[0].message
    messages.append(assistant_msg)

    if not assistant_msg.tool_calls:
        break

    for tool_call in assistant_msg.tool_calls:
        result = run_tool(tool_call)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
```

这也是实现 ReAct agent 的基础。百炼支持 Function Calling，但不会替你把整个多步 Agent Loop 自动编排完。

---

## 5. 并行工具调用

## 5.1 能力说明

如果一个请求包含多个**互不依赖**的工具调用任务，可以启用 `parallel_tool_calls=True`，让模型在一次响应里返回多个 `tool_calls`，例如：

- “北京和上海天气如何”
- “杭州天气，以及现在几点了”

如果不开启并行工具调用，模型可能只先返回一个工具调用。

## 5.2 Python 示例

```python
completion = client.chat.completions.create(
    model="qwen3.6-plus",
    extra_body={"enable_thinking": False},
    messages=messages,
    tools=tools,
    parallel_tool_calls=True,
)
```

随后，应用端可并行执行 `tool_calls` 数组中的多个任务，再统一回填结果。

## 5.3 何时适用

适用于：

- 多个工具任务之间没有依赖
- 工具执行时间较长，适合并发
- 任务可以独立失败和独立返回

不适用于：

- B 工具依赖 A 工具结果
- 需要模型先看第一个结果再决定第二个动作
- 需要严格的串行推理链

对于 Deep Research Agent：

- 多个互不依赖的网页 HEAD 请求、元信息获取、多个城市天气等，适合并行
- 搜索 -> 选页 -> 抓正文 -> 摘要 -> 证据判断 这类依赖链，不适合单轮并行，仍应由应用端串行控制

---

## 6. tool_choice 控制

百炼支持通过 `tool_choice` 控制工具调用策略。

## 6.1 默认自动选择

```python
tool_choice="auto"
```

通常可以省略，由模型自行判断是否调用工具以及调用哪个工具。

## 6.2 强制调用某个工具

```python
tool_choice={
    "type": "function",
    "function": {"name": "get_current_weather"}
}
```

适用于：

- 某类请求必须走某个特定工具
- 业务上不希望模型在多个工具间自行选择
- 已经由上层路由器决定好了工具

注意：强制工具调用时，模型不再负责选工具，只会生成该工具的入参。如果问题与该工具不相关，可能得到不合理入参。

## 6.3 强制不调用工具

```python
tool_choice="none"
```

适用于：

- 希望模型直接回答
- 在第二次总结工具结果时，不希望再次触发工具调用
- 某些阶段只允许纯文本输出

工程上尤其要注意：**当模型在总结工具输出时，应去掉强制工具调用设置，否则 API 仍可能继续返回工具调用信息。**

---

## 7. 消息结构要求

在多轮工具调用场景下，建议维护完整的 `messages` 历史。一个典型结构如下：

```python
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "用户问题"},
  {"role": "assistant", "content": "", "tool_calls": [...]},
  {"role": "tool", "tool_call_id": "...", "content": "工具输出"},
  {"role": "assistant", "content": "模型基于工具输出的总结"},
  {"role": "user", "content": "下一轮问题"}
]
```

对于工程日志系统，应完整记录：

- 本轮模型可见的 messages
- assistant 返回的 `tool_calls`
- tool 执行参数和原始结果
- 追加后的 messages 变化

---

## 8. 百炼能力边界与 Agent 实现建议

## 8.1 百炼已经提供的能力

百炼官方能力主要包括：

- OpenAI 兼容 Chat Completions
- `tools` / `tool_calls` Function Calling 机制
- `parallel_tool_calls`
- `tool_choice`
- `enable_thinking`
- 流式输出下的 `reasoning_content` / `content`

## 8.2 百炼不会替你完成的部分

以下能力应由业务侧 Runtime / Harness 自行实现：

- ReAct Agent Loop（Thought -> Act -> Observe -> Thought -> Final）
- 工具实际执行
- 工具权限控制和异常重试
- 上下文裁剪与外部工作记忆管理
- 文件归档
- TODO 状态机
- 可重放日志
- 搜索结果筛选与正文抓取编排

换句话说，百炼提供的是“模型调用能力”和“Function Calling 协议能力”，不是完整 Agent Runtime。

---

## 9. 面向本项目的推荐实现方式

结合 Deep Research Agent 的需求，推荐采用如下策略：

### 9.1 模型层

使用百炼 OpenAI 兼容接口作为统一模型接入层。

### 9.2 工具层

所有工具由业务侧自行定义、注册和执行，不依赖百炼内置工具。

### 9.3 Agent Loop

采用应用端 while-loop 驱动的 ReAct 模式：

1. 组装上下文
2. 发起模型调用
3. 读取显式思考 / 输出 / tool_calls
4. 执行工具
5. 写入日志与状态快照
6. 回填工具结果
7. 再次思考与继续执行
8. 直到返回最终文本

### 9.4 并行工具调用

对于完全独立的工具任务，可开启 `parallel_tool_calls=True`，但仍由应用端负责并发执行和回填。

### 9.5 日志与可重放

应额外实现 Runtime Logging：

- 记录本轮输入 messages
- 记录模型显式思考内容
- 记录 tool_calls 与 tool results
- 记录上下文变化
- 记录状态对象变化

---

## 10. 最小接入建议

如果要最快落地一个可运行版本，建议按以下顺序接入：

1. 打通 OpenAI 兼容普通对话调用
2. 打通单工具 Function Calling
3. 实现应用端串行 while-loop
4. 加入 `enable_thinking` 与流式思考日志
5. 加入 `parallel_tool_calls=True` 的并发执行
6. 最后再接入完整的 research/writing 上下文管理与日志系统

---

## 11. 常见实现误区

### 误区 1：以为开启 enable_thinking 就等于实现了 Agent 思考循环

不是。`enable_thinking` 只影响模型本次回复前的显式思考输出，不会替你完成“工具调用后再思考、再决定下一步”的运行时循环。

### 误区 2：以为 Function Calling 会自动执行工具

不是。模型只会返回工具名称和入参，真正执行必须由应用端完成。

### 误区 3：以为 parallel_tool_calls 可以覆盖所有多工具场景

不是。它只适合任务间无依赖的情况。有依赖的多步任务仍要用应用端 while-loop 串行执行。

### 误区 4：在工具结果总结阶段仍强制 tool_choice

这样可能导致模型继续输出工具调用而不是生成总结文本。总结阶段通常应移除强制工具选择，或显式设为 `none`。

---

## 12. 结论

对于本项目，百炼最适合扮演的角色是：

- 提供稳定的模型推理能力
- 提供 OpenAI 兼容的 Function Calling 协议
- 提供思考模式、并行工具调用和工具选择控制

而真正的 Agent 系统能力——包括 ReAct loop、工具执行、上下文管理、文件归档、日志系统、TODO 状态机——应由我们自己的 Runtime / Harness 层来实现。
