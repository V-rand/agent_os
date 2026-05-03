# Hermes Agent 架构工程蓝图

> **摘要**：本文档详细拆解了 Hermes Agent 的底层运行时（Runtime）架构。设计哲学是将"底层运行系统"（稳健、可控、高性能）与"上层工作流"解耦，采用 **Sync-First** 架构模式，通过精细的上下文管理、双轨记忆系统和递归子智能体实现复杂任务处理。

---

## 1. 核心设计哲学 (Core Philosophy)

*   **Sync-First 模型**：顶层 `run_conversation` 是**同步阻塞**的，底层通过 `ThreadPoolExecutor` 处理工具并发。放弃全 Async Loop 以降低心智负担，防止死锁并简化调试。
*   **ReAct 循环**：遵循 `Thought -> Action -> Observation` 的标准范式，严格限制最大迭代次数（`max_iterations`）防止无限循环。
*   **KV Cache 亲和性**：所有 Prompt 构建和消息组装都围绕 **"Prefix Stability"（前缀稳定性）** 设计，确保在长对话中最大化复用 KV Cache，降低 Token 成本。

---

## 2. 工具系统架构 (The Tool System)

工具不仅仅是函数，是 Agent 与环境交互的标准化接口。

### 2.1 注册与自动发现 (`tools/registry.py`)
*   **AST 静态扫描**：系统启动时，通过 AST 扫描 `tools/*.py`，自动发现所有注册的工具，无需手动导入。
*   **Schema 自动生成**：工具定义包含 JSON Schema，系统自动将其转换为 OpenAI 兼容的格式。
*   **动态门控 (`check_fn`)**：每个工具可注册一个 `check_fn`。
    *   *逻辑*：如果 `check_fn` 返回 `False`（如缺少 API Key 或未安装依赖），该工具不仅不会被执行，其 **Schema 也会被从 API 请求中移除**，防止模型产生幻觉调用不存在的工具。

### 2.2 工具集管理 (Toolsets)
*   **分组控制**：工具被划分为 Toolsets（如 `web`, `terminal`, `files`）。
*   **动态过滤**：用户在配置中启用/禁用 Toolset，系统会实时更新注入给模型的可用工具列表。

---

## 3. 系统提示词构建 (The Prompt Builder)

System Prompt 不是静态文本，而是一个**动态组装的引擎** (`agent/prompt_builder.py`)。

### 3.1 五层拼装结构
按优先级从高到低组装：
1.  **Identity (身份)**：加载 `SOUL.md` 或默认身份。
2.  **Behavioral Patches (行为补丁)**：
    *   *模型级*：根据模型名称（如 `gpt-4` vs `gemini`）注入特定的"执行纪律"提示词。
    *   *平台级*：根据平台（CLI, Telegram, Discord）注入格式指令（如"不要用 Markdown"）。
3.  **Memory Snapshot (记忆快照)**：注入从 `MEMORY.md` 读取的用户偏好和环境事实。
4.  **Skill Index (技能索引)**：注入可用技能的"目录"（名称+简介），而非全文。
5.  **Context Files (项目上下文)**：按需加载 `AGENTS.md`, `.cursorrules` 等，注入前进行**安全扫描**（防 Prompt 注入）。

### 3.2 关键工程细节：技能注入 (Skill Injection)
*   **菜单模式**：System Prompt 只包含 Skills 的索引（`<available_skills>`）。
*   **按需加载**：模型通过调用 `skill_view` 工具获取具体技能的全文。这避免了 Prompt 爆炸，同时保证了上下文的紧凑。
*   **过滤机制**：如果技能依赖的工具（如 `docker`）未启用，该技能会自动从索引中隐藏。

---

## 4. 上下文与状态管理 (Context & State)

### 4.1 双轨记忆系统 (Dual-Track Memory)
*   **轨道 A：内置记忆 (File-Based)**
    *   *实现*：`tools/memory_tool.py` 中的 `MemoryStore` 类。
    *   *存储*：`MEMORY.md`（环境事实）和 `USER.md`（用户偏好）。
    *   *快照机制*：**关键设计！** 会话开始时生成 `_system_prompt_snapshot`，全程**只读**。
    *   *写入*：Agent 调用 `memory` 工具写入新记忆，直接修改磁盘文件。**当前会话的 System Prompt 不会变**（保护 KV Cache），新记忆通过工具返回值告知 Agent。
*   **轨道 B：外部记忆 (Plugin-Based)**
    *   *实现*：`agent/memory_manager.py` 协调插件（如 Mem0, Honcho）。
    *   *注入*：通过 `prefetch` 在每一轮动态检索，注入到 **User Message** 而非 System Prompt。

### 4.2 任务状态 (State Management)
*   **TodoStore**：基于内存的列表。Agent 通过 `todo` 工具更新进度。
*   **压缩防丢**：当上下文触发压缩时，系统强制将"未完成"的 Todo 提取出来，作为纯文本追加到压缩后的历史中，防止 Agent 失忆。

### 4.3 会话持久化
*   **SQLite**：所有对话历史（History）存入 SQLite。
*   **Session Split**：当上下文过长触发压缩时，旧 Session 被标记为 `ended`，新 Session 带着摘要开始，通过 `parent_session_id` 保持血缘关系。

---

## 5. 子智能体架构 (Subagent Orchestration)

解决复杂任务的核心机制 (`tools/delegate_tool.py`)。

### 5.1 递归实例化
*   **实现**：子智能体实际上是父 Agent 代码库中另一个全新的 `AIAgent` 实例。
*   **隔离**：
    *   全新的 `session_id`。
    *   独立的 `messages` 列表（不继承父级历史）。
    *   独立的 `task_id`（隔离终端会话和文件缓存）。

### 5.2 角色与权限
*   **Leaf（叶子）**：默认角色。只能干活，**被禁止**调用 `delegate_task`（防止无限套娃）、`memory`（防止写乱主记忆）或 `clarify`（不能打扰用户）。
*   **Orchestrator（协调者）**：拥有分发任务的权限，可以管理下级 Agent，受最大深度（`MAX_DEPTH`）限制。

### 5.3 执行流
*   父 Agent 使用 `ThreadPoolExecutor` 并发启动子 Agent。
*   父 Agent 阻塞等待结果，期间显示 "Waiting" 动画。
*   结果以 JSON 形式返回，不暴露中间过程，保持上下文整洁。

---

## 6. 基础设施与稳健性 (Infrastructure)

### 6.1 日志系统 (`hermes_logging.py`)
*   **会话上下文注入**：通过替换 Python 全局 `LogRecordFactory`，将 `[session_id]` 强制注入每一条日志，实现多会话并发时的日志隔离与检索。
*   **敏感信息脱敏**：使用 `RedactingFormatter`，在写入磁盘前通过正则拦截并替换 API Key，防止密钥泄露。

### 6.2 文件系统操作
*   **原子写入**：修改文件时先写临时文件，再原子重命名（Atomic Replace），防止 Agent 崩溃导致文件损坏。
*   **文件锁**：在多子 Agent 并发修改同一文件时，使用 `fcntl` 锁防止冲突。

### 6.3 安全扫描
*   在加载 `AGENTS.md` 或写入记忆时，扫描恶意指令（如 "Ignore previous instructions", "Read secrets"）和隐藏 Unicode 字符，防止 Prompt 注入攻击。

---

## 7. 复现清单 (Checklist for Implementation)

如果您要复现此系统，请重点关注以下实现顺序：

1.  **Registry**：实现 `registry.register()` 和基于 AST 的自动发现。
2.  **Toolsets**：实现工具分组和 Schema 过滤逻辑。
3.  **Sync Loop**：编写基于 `while` 的 ReAct 循环，处理 `client.chat.completions.create`。
4.  **Dispatcher**：实现 `handle_function_call`，通过线程池桥接异步工具。
5.  **Prompt Builder**：实现分层拼装，**务必实现 System Prompt 的快照机制**。
6.  **Memory Store**：实现 `MemoryStore`，分离“快照读取”和“实时写入”。
7.  **Delegation**：实现 `delegate_task`，封装新的 Agent 实例并处理线程隔离。
8.  **Logging**：实现 Session Tag 注入和 Secret 过滤。
