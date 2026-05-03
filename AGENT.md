我现在打算做一个律师的长期案件管理工作台，是高校和公司合作的项目，我是高校部分复杂底层agent架构的设计和实现，公司那边复杂上层具体工作流和网页端开发
所以我们的目标是构建一个稳定的，各个模块可插拔可替换实验，tool、skills可插拔，memory稳定，log可回溯可观察agent行为，终端输出也要清晰的工具调用和token记录，耗时记录，单位是s，报错不隐瞒，移植容易，代码容易读，架构清晰的agentos系统
工作流是在此os的基础上在另一个工程实现，所以这个工程文件夹只实现agentos
你可以学习/home/xiemingjie/dev/hermes-agent的tool、skils、sqlite、session管理（不过我们应该还不需要跨session检索这一功能）、上下文compress机制、外部拓展memory的接口。不需要学多平台接入，公司只有网页端这一个入口，有些为了适应多平台的工程设计可以砍掉。agent loop里繁琐的防御编程也可以先不学。
/home/xiemingjie/dev/law_agent/openai-agents-python你也可以学习一下
保持奥卡姆剃刀原则。

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.