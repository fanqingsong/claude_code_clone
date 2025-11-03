# Claude-Code-Clone — LangGraph CLI Coding agent

A compact, runnable Python project that reconstructs a demo agent using LangGraph, LangChain and Anthropic Claude. The project provides a terminal UI (Rich), local utility tools, and support for remote MCP servers. This README focuses on getting started with uv and common workflows.

## Key features
- Interactive agent driven by a state graph (user input → model response → tool use → back to user).
- Local tools: file reader and unit-test runner (Pytest wrapper).
- MCP integrations (DesktopCommander, sandbox Python MCP, DuckDuckGo search, GitHub MCP, and a Deno Docker image).
- Rich terminal UI and Mermaid workflow visualization.

## Prerequisites
- macOS / Linux / Windows with Python 3.11+ (project uses 3.13 bytecode in cache but is compatible with 3.11+).
- uv
- Docker (required to build/run the provided MCP Docker images- ensure that Docker Desktop is running).

## Quick start (using uv)
1. Initialize the uv workspace (creates .venv and metadata):

   uv init

2. Install dependencies from requirements.txt into the uv-managed venv:

   uv add -r requirements.txt

3. Sync uv's lock state (optional but recommended):

   uv sync

4. Activate the virtual environment created by uv (common path):

   source .venv/bin/activate

5. Run the agent CLI:

   uv run main.py

You can also run directly with Python if you prefer (after activating venv):

   python3 main.py

## 交互流程说明 (How It Works)

### 一问一答的循环机制

本项目是一个 CLI 对话式代码助手，实现了用户与智能体之间的持续交互。**循环逻辑并不是通过显式的 Python `while` 循环实现的**，而是通过 **LangGraph 工作流图的结构**自动实现的。

### 工作流图结构

工作流由三个核心节点组成，通过有向边连接：

```
┌─────────────┐
│ user_input  │ ←──┐ (等待下一轮输入)
└──────┬──────┘    │
       │           │
       ▼           │
┌─────────────┐   │
│model_response│ ──┘ (无工具调用时)
└──────┬──────┘
       │
       │ (条件路由)
       ├───→ 有工具调用 ──→ ┌──────────┐
       │                  │ tool_use │
       │                  └────┬─────┘
       │                       │
       └─── 无工具调用           │
                              ▼
                        ┌─────────────┐
                        │model_response│ (处理工具结果)
                        └─────────────┘
```

### 详细执行流程

1. **启动阶段** (`main.py`)
   - 程序启动时调用 `agent.run()`，这会触发工作流的第一次执行
   - 传入初始消息："What can I do for you?"

2. **用户输入节点** (`user_input`)
   - 程序在终端显示 `> ` 提示符，**阻塞等待用户输入**
   - 用户输入内容后，封装为 `HumanMessage` 添加到状态中
   - 自动流转到下一个节点

3. **模型响应节点** (`model_response`)
   - 调用 Claude 模型（已绑定工具），生成响应
   - 响应可能包含：
     - **纯文本回答**：直接显示给用户
     - **工具调用请求**：模型决定需要调用工具来完成任务
   - 响应会显示在终端（使用 Rich 库美化）

4. **条件路由判断** (`check_tool_use`)
   - 检查模型响应是否包含 `tool_calls`
   - **如果有工具调用** → 路由到 `tool_use` 节点
   - **如果没有工具调用** → 路由回 `user_input` 节点 ⬅️ **形成循环**

5. **工具调用节点** (`tool_use`) [可选]
   - 如果模型决定调用工具，执行相应的工具（如运行测试、搜索、读取文件等）
   - 工具执行结果封装为 `ToolMessage`
   - 执行完成后，**自动流转回 `model_response` 节点**，让模型处理工具结果

6. **循环继续**
   - 模型处理完工具结果后，再次进行路由判断
   - 如果模型认为还需要更多工具调用，会再次进入 `tool_use`
   - 如果模型已经完成了任务，会路由回 `user_input`，等待用户下一轮输入
   - 这个循环会**持续进行，直到用户中断程序**（Ctrl+C）或发生异常

### 关键设计特点

- **基于图结构的循环**：循环通过工作流图的边和节点的阻塞行为（`user_input` 节点的 `input()` 调用会阻塞等待用户输入）实现，而不是显式的 Python `while` 循环代码
- **状态持久化**：所有对话历史都保存在 SQLite 数据库中（`checkpoints.db`），支持跨会话的上下文记忆
- **灵活的决策**：模型可以自主决定是否需要调用工具，也可以直接回答用户问题
- **优雅的错误处理**：工具调用失败时会返回错误消息给模型，模型可以基于错误信息调整策略

### 示例对话流程

```
[启动] 
  → Agent: "What can I do for you?"
  
[用户输入节点] 
  → 用户输入: "帮我运行测试"
  
[模型响应节点] 
  → Agent: "我来帮您运行测试..."
  → Agent 决定调用工具: run_unit_tests
  
[工具调用节点] 
  → 🔧 执行 run_unit_tests
  → 返回测试结果
  
[模型响应节点] (处理工具结果)
  → Agent: "测试运行完成，共 10 个测试，全部通过！"
  
[路由判断] 
  → 无更多工具调用
  
[用户输入节点] ⬅️ 回到这里，等待下一轮输入
  → 用户输入: "谢谢"
  
[模型响应节点]
  → Agent: "不客气，还有什么可以帮您的吗？"
  
[循环继续...]
```

### 为什么这样设计？

这种基于图结构的循环设计有以下优势：

1. **可视化**：工作流图可以生成 Mermaid 图表，直观展示整个流程
2. **可扩展**：容易添加新的节点（如人工审批、日志记录等）
3. **状态管理**：LangGraph 自动管理状态流转和持久化
4. **错误恢复**：支持检查点机制，可以从中间状态恢复

## Environment variables (.env)
Create a .env file in the project root or export env vars before running.
Example .env:

  ANTHROPIC_API_KEY=sk-ant-...
  GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...


## Useful uv commands and examples
- Run the main agent:
  uv run main.py

- Build the Deno MCP Docker image:
  docker build -t deno-docker:latest -f ./mcps/deno/Dockerfile .

## Common prompts to try
- summarize the recent articles from https://simonwillison.net/
- use python_run_code tool to run ascii_art_generator.py
- "Show me the content of main.py" (assuming you have exposed this to Desktop Commander MCP or enable built-in read_file tool)
- "What tools do you have?"
- "Read /absolute/path/to/requirements.txt"

## Available tools and MCPs
Local tools (bundled in tools/):
- file_read_tool.py — safely reads and returns file contents; handles permission and not-found errors. Not used because we decided to use Desktop Commander MCP instead
- run_unit_tests_tool.py — wrapper that runs pytest and returns results.

- Run a local tool (file reader):
  uv run tools/file_read_tool.py -- /absolute/path/to/file.txt

  (The file reader will print contents and handle common file errors.)

- Run unit-test runner (project provides a Pytest wrapper):
  uv run tools/run_unit_tests_tool.py

Remote MCPs (configured in repo):
- DesktopCommander MCP
- Pydantic AI run-python (sandbox Python MCP)
- DuckDuckGo search MCP
- GitHub MCP (runs as a Docker container; requires GITHUB_PERSONAL_ACCESS_TOKEN)
    ```
    command: docker 
    Arguments: run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=GITHUB_PERSONAL_ACCESS_TOKEN ghcr.io/github/github-mcp-server
    ```

## Inspecting the SQLite database
The project uses SQLite to store checkpoints. You can inspect the database using the sqlite3 command-line tool:

   sqlite3 checkpoints.db

Common SQLite commands:
- List all tables:
  .tables

- Show table schema:
  .schema your_table_name

- Export query results:
  .mode csv
  .output results.csv
  .headers on
  SELECT * FROM your_table_name;
  .output stdout

Exit sqlite3 with .quit or Ctrl+D

## Development notes
- The agent composes system + working-directory guidance to the Claude model. You can change model parameters in the code if you prefer a different LLM.
- Tools are designed to return structured ToolMessages so the StateGraph can route responses back to the model correctly.
- The terminal UI uses Rich for Markdown, code highlighting, and Mermaid output.

## Troubleshooting
- uv: If `uv run` fails, ensure you ran `uv init` and `uv add -r requirements.txt`, and that you activated the .venv.
- Missing API key: set ANTHROPIC_API_KEY in .env or export it before running.
- Docker errors: verify Docker is running and you have permission to run docker commands.
- Python version mismatch: use the Python version your virtual environment is created with; recreate the venv if needed.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security
- This project reads files but does not execute arbitrary shell commands or user files. Review tools before trusting them with sensitive directories.