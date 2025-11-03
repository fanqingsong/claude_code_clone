"""
Agent 主模块
基于 LangGraph 实现的对话式代码助手，支持工具调用和 MCP 集成
"""
from typing import Annotated, Sequence
from dotenv import load_dotenv
import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langgraph.graph.message import add_messages
from tools.run_unit_tests_tool import run_unit_tests
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# import sqlite3
# import aiosqlite


class AgentState(BaseModel):
    """
    工作流状态类
    在工作流图的各个节点之间持久化传递的状态
    
    Attributes:
        messages: 完整的聊天历史，包含系统消息、用户消息、助手消息和工具消息
                  使用 add_messages 合并函数来自动处理消息列表的合并
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]


class Agent:
    """
    Agent 主类
    负责管理对话式代码助手的工作流、工具调用和用户交互
    
    工作流图结构：
    - user_input: 获取用户输入
    - model_response: 调用 LLM 生成响应
    - tool_use: 执行工具调用（如果 LLM 决定使用工具）
    - 条件路由: 根据 LLM 响应是否包含工具调用，路由到 tool_use 或回到 user_input
    """
    
    def __init__(self):
        """
        初始化 Agent 实例
        加载环境变量、创建 LLM 模型、初始化工作流图结构
        """
        self._initialized = False
        
        # 加载环境变量（从 .env 文件）
        load_dotenv()
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing ANTHROPIC_API_KEY in environment. Set it in .env or your shell."
            )

        # 实例化 Claude 模型（Claude Sonnet 最新版本）
        # temperature=0.3: 较低的温度值，使输出更加一致和可预测
        # max_tokens=4096: 限制最大输出长度
        self.model = ChatAnthropic(
            model="claude-3-7-sonnet-latest",
            temperature=0.3,
            max_tokens=4096,
            api_key=api_key,
        )

        # Rich 控制台：用于美化的终端输出（彩色、面板、Markdown 渲染等）
        self.console = Console()

        # 创建工作流图：使用 StateGraph 管理状态流转
        self.workflow = StateGraph(AgentState)

        # 注册工作流节点
        # 节点是工作流中的处理单元，每个节点负责特定的任务
        self.workflow.add_node("user_input", self.user_input)      # 获取用户输入
        self.workflow.add_node("model_response", self.model_response)  # 生成模型响应
        self.workflow.add_node("tool_use", self.tool_use)          # 执行工具调用

        # 设置工作流边（Edge）：定义节点之间的流转关系
        # 入口节点：工作流从 user_input 开始
        self.workflow.set_entry_point("user_input")
        # 固定边：user_input 总是流向 model_response
        self.workflow.add_edge("user_input", "model_response")
        # 固定边：tool_use 执行完成后，回到 model_response 让模型处理工具结果
        self.workflow.add_edge("tool_use", "model_response")

        # 条件边：根据模型响应决定下一步路由
        # - 如果响应包含工具调用，路由到 tool_use
        # - 如果没有工具调用，路由回 user_input 等待下一轮用户输入
        self.workflow.add_conditional_edges(
            "model_response",
            self.check_tool_use,  # 路由判断函数
            {
                "tool_use": "tool_use",
                "user_input": "user_input",
            },
        )

    async def initialize(self):
        """
        异步初始化方法
        加载工具（本地工具和 MCP 工具）、绑定工具到模型、编译工作流图
        
        Returns:
            self: 返回自身以便链式调用
        """
        # 防止重复初始化
        if self._initialized:
            return self

        print("🔄 Initializing agent...")

        # 加载本地工具
        # run_unit_tests: 运行单元测试的工具
        local_tools = [run_unit_tests]

        # 设置 MCP (Model Context Protocol) 客户端并获取 MCP 工具
        # MCP 是一个协议，允许 Agent 通过 Docker 容器访问外部服务（如 GitHub、DuckDuckGo 等）
        mcp_tools = await self.get_mcp_tools()
        self.tools = local_tools + mcp_tools
        print(
            f"✅ Loaded {len(self.tools)} total tools (Local: {len(local_tools)} + MCP: {len(mcp_tools)})"
        )
        self._initialized = True

        # 将工具绑定到模型
        # bind_tools 使模型能够理解和调用这些工具
        self.model_with_tools = self.model.bind_tools(self.tools)

        # 编译工作流图：创建可执行的工作流实例
        # 使用 SQLite 检查点（checkpointer）持久化对话状态
        # 注意：这里先使用临时上下文管理器创建了一个实例（已注释掉），
        # 然后手动管理 AsyncSqliteSaver 的生命周期，避免重复打开/关闭数据库连接
        # 这样可以保持数据库连接在整个 Agent 生命周期内保持打开状态，提高性能
        db_path = os.path.join(os.getcwd(), "checkpoints.db")
        self._checkpointer_ctx = AsyncSqliteSaver.from_conn_string(db_path)
        self.checkpointer = await self._checkpointer_ctx.__aenter__()
        # 编译工作流图，传入检查点管理器以支持状态持久化
        self.agent = self.workflow.compile(checkpointer=self.checkpointer)

        # Optional: print a greeting panel
        self.console.print(
            Panel.fit(
                Markdown("**LangGraph Coding Agent** — Claude Code Clone"),
                title="[bold green]Ready[/bold green]",
                border_style="green",
            )
        )
        return self

    async def run(self):
        """
        启动工作流的主循环
        
        注意：这个方法只触发工作流的第一次执行。
        实际的循环逻辑是通过工作流图的结构实现的：
        - user_input -> model_response -> (tool_use -> model_response 或 -> user_input)
        - 当模型不需要调用工具时，会路由回 user_input 节点，形成循环
        - 循环会持续进行，直到程序被用户中断（Ctrl+C）或异常退出
        
        Returns:
            工作流的执行结果（通常是最后一次状态）
        """
        # 配置工作流执行的线程 ID
        # thread_id 用于区分不同的对话会话，同一个 thread_id 会共享检查点状态
        config = {"configurable": {"thread_id": "1"}}
        
        # 触发工作流执行：从初始的助手消息开始
        # 这个初始消息会启动整个工作流循环
        return await self.agent.ainvoke(
            {"messages": AIMessage(content="What can I do for you?")}, config=config
        )

    async def close_checkpointer(self):
        """
        关闭异步检查点上下文
        
        清理资源：关闭 SQLite 数据库连接，确保数据正确保存
        应在 Agent 生命周期结束时调用（通常在 main.py 的退出清理阶段）
        """
        if hasattr(self, "_checkpointer_ctx"):
            await self._checkpointer_ctx.__aexit__(None, None, None)

    async def get_mcp_tools(self):
        """
        获取 MCP (Model Context Protocol) 工具
        
        通过 Docker 容器运行多个 MCP 服务器，每个服务器提供不同的工具能力：
        - Run_Python_MCP: 在 Deno 环境中运行 Python 代码
        - duckduckgo_MCP: 提供网络搜索功能
        - desktop_commander_in_docker_MCP: 提供桌面命令执行能力（已挂载文档目录）
        - Github_MCP: 提供 GitHub 操作能力（需要访问令牌）
        
        Returns:
            List: MCP 工具列表，可以绑定到 LLM 供其调用
        """
        from langchain_mcp_adapters.client import MultiServerMCPClient

        # 获取 GitHub 访问令牌（用于 GitHub MCP 服务器）
        GITHUB_PERSONAL_ACCESS_TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
        
        # 创建多服务器 MCP 客户端
        # 每个服务器通过 Docker 容器运行，使用 stdio 传输协议进行通信
        mcp_client = MultiServerMCPClient(
            {
                "Run_Python_MCP": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",      # 交互式模式
                        "--rm",    # 容器退出后自动删除
                        "deno-docker:latest",  # Deno 运行时镜像
                        "deno",    # 在容器内执行的命令
                        "run",
                        "-N",      # 允许网络访问
                        "-R=node_modules",
                        "-W=node_modules",
                        "--node-modules-dir=auto",
                        "jsr:@pydantic/mcp-run-python",  # MCP 服务器包
                        "stdio",   # 使用标准输入输出通信
                    ],
                    "transport": "stdio",
                },
                "duckduckgo_MCP": {
                    "command": "docker",
                    "args": ["run", "-i", "--rm", "mcp/duckduckgo"],
                    "transport": "stdio",
                },
                "desktop_commander_in_docker_MCP": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-v",  # 挂载卷
                        "/Users/lorreatlan/Documents/MyPlayDocuments:/mnt/documents",
                        "mcp/desktop-commander:latest",
                    ],
                    "transport": "stdio",
                },
                "Github_MCP": {
                    "command": "docker",
                    "args": [
                        "run",
                        "-i",
                        "--rm",
                        "-e",  # 设置环境变量
                        f"GITHUB_PERSONAL_ACCESS_TOKEN={GITHUB_PERSONAL_ACCESS_TOKEN}",
                        "-e",
                        "GITHUB_READ-ONLY=1",  # 只读模式
                        "ghcr.io/github/github-mcp-server",
                    ],
                    "transport": "stdio",
                },
            }
        )
        # 从所有 MCP 服务器获取工具列表
        mcp_tools = await mcp_client.get_tools()
        # 打印所有可用的 MCP 工具名称
        for tb in mcp_tools:
            print(f"MCP 🔧 {tb.name}")
        return mcp_tools

    # Node: user_input
    def user_input(self, state: AgentState) -> AgentState:
        """
        工作流节点：获取用户输入
        
        提示用户输入，将用户输入封装为 HumanMessage 并添加到状态中。
        这是工作流循环的起点，每次模型完成响应后（不需要工具调用时）会回到这里。
        
        Args:
            state: 当前工作流状态，包含之前的消息历史
        
        Returns:
            AgentState: 更新后的状态，包含新的用户消息
        """
        self.console.print("[bold cyan]User Input[/bold cyan]: ")
        user_input = self.console.input("> ")
        return {"messages": [HumanMessage(content=user_input)]}

    # Node: model_response
    def model_response(self, state: AgentState) -> AgentState:
        """
        工作流节点：生成模型响应
        
        调用绑定了工具的 LLM 模型，生成响应。响应可能包含：
        1. 普通文本回答
        2. 工具调用请求（如果需要执行工具）
        
        响应会被打印到控制台，然后根据是否包含工具调用路由到下一个节点。
        
        Args:
            state: 当前工作流状态，包含完整的对话历史
        
        Returns:
            AgentState: 更新后的状态，包含模型的响应消息
        """
        # 系统提示词：定义 Agent 的行为准则和开发规范
        system_text = """You are a specialised agent for maintaining and developing codebases.
            ## Development Guidelines:

            1. **Test Failures:**
            - When tests fail, fix the implementation first, not the tests.
            - Tests represent expected behavior; implementation should conform to tests
            - Only modify tests if they clearly don't match specifications

            2. **Code Changes:**
            - Make the smallest possible changes to fix issues
            - Focus on fixing the specific problem rather than rewriting large portions
            - Add unit tests for all new functionality before implementing it

            3. **Best Practices:**
            - Keep functions small with a single responsibility
            - Implement proper error handling with appropriate exceptions
            - Be mindful of configuration dependencies in tests

            Ask for clarification when needed. Remember to examine test failure messages carefully to understand the root cause before making any changes."""
        
        # 组合消息列表：系统消息 + 当前工作目录提示 + 历史对话消息
        # 系统消息使用 ephemeral 缓存控制，表示这是临时性的提示，不应该被持久化缓存
        messages = [
            SystemMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            ),
            HumanMessage(content=f"Working directory: {os.getcwd()}"),
        ] + state.messages

        # 调用模型：使用绑定了工具的模型生成响应
        response = self.model_with_tools.invoke(messages)
        
        # response 的数据格式说明：
        # 1. 普通文本响应:
        # response.content 可能为:
        # [
        #   {"type": "text", "text": "你好，有什么可以帮助您的？"}
        # ]
        # 2. 包含工具调用:
        # response.content 可能为:
        # [
        #   {"type": "text", "text": "正在帮您执行工具调用..."},
        #   {"type": "tool_use", "name": "run_tests", "args": {"test_path": "tests/test_agent.py"}, "id": "call_1"}
        # ]
        # 3. 少数情况下也可能直接是字符串:
        # response.content == "Assistant message content here."
        # 4. response 还可能有 tool_calls 属性:
        # response.tool_calls = [
        #   {"name": "run_tests", "args": {"test_path": "tests/test_agent.py"}, "id": "call_1"}
        # ]
        if isinstance(response.content, list):
            # 处理列表格式的响应（包含多个内容块）
            for item in response.content:
                if item["type"] == "text":
                    # 显示文本响应：使用 Markdown 格式渲染
                    text = item.get("text", "")
                    if text:
                        self.console.print(
                            Panel.fit(
                                Markdown(text),
                                title="[magenta]Assistant[/magenta]",
                                border_style="magenta",
                            )
                        )
                elif item["type"] == "tool_use":
                    # 显示工具调用预览：显示即将调用的工具名称和参数
                    self.console.print(
                        Panel.fit(
                            Markdown(
                                f"{item["name"]} with args {item.get("args",None)}"
                            ),
                            title="Tool Use",
                        )
                    )
        else:
            # 处理字符串格式的响应（简单文本）
            self.console.print(
                Panel.fit(
                    Markdown(response.content),
                    title="[magenta]Assistant[/magenta]",
                )
            )

        # 返回更新后的状态：将模型响应添加到消息历史中
        return {"messages": [response]}

    # Conditional router
    def check_tool_use(self, state: AgentState) -> str:
        """
        条件路由函数：决定工作流的下一步
        
        检查最后一条助手消息是否包含工具调用：
        - 如果有工具调用：路由到 "tool_use" 节点执行工具
        - 如果没有工具调用：路由到 "user_input" 节点等待下一轮用户输入
        
        Args:
            state: 当前工作流状态
        
        Returns:
            str: 下一个节点的名称（"tool_use" 或 "user_input"）
        """
        if state.messages[-1].tool_calls:
            return "tool_use"
            
        return "user_input"

    # Node: tool_use
    async def tool_use(self, state: AgentState) -> AgentState:
        """
        工作流节点：执行工具调用
        
        从最后一条助手消息中提取工具调用请求，执行每个工具调用，
        并将结果封装为 ToolMessage 返回。工具结果会保留 tool_call_id，
        以便模型在处理结果时能够正确关联。
        
        执行流程：
        1. 遍历所有工具调用请求
        2. 查找对应的工具实例
        3. 使用 ToolNode 执行工具
        4. 捕获异常并返回错误消息
        5. 所有工具结果会返回到 model_response 节点，让模型处理结果
        
        Args:
            state: 当前工作流状态，最后一条消息应包含 tool_calls
        
        Returns:
            AgentState: 更新后的状态，包含工具执行的 ToolMessage 结果
        """
        from langgraph.prebuilt import ToolNode

        response = []
        # 创建工具名称到工具实例的映射，便于快速查找
        tools_by_name = {t.name: t for t in self.tools}

        # 遍历所有工具调用请求
        for tc in state.messages[-1].tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]
            print(f"🔧 Invoking tool '{tool_name}' with args {tool_args}")
            tool = tools_by_name.get(tool_name)
            print(f"🛠️ Found tool: {tool}")
            
            # 使用 ToolNode 包装工具，以便在工作流中执行
            tool_node = ToolNode([tool])

            # 注释掉的代码：工具调用前的审批机制示例
            # 可以用于在生产环境中添加人工审核步骤
            # response = interrupt(
            #     {
            #         "action": "review_tool_call",
            #         "tool_name": tool_name,
            #         "tool_input": state["messages"][-1].content,
            #         "message": "Approve this tool call?",
            #     }
            # )
            # # Handle the response after the interrupt (e.g., resume or modify)
            # if response == "approved":
            try:
                # 执行工具调用
                tool_result = await tool_node.ainvoke(state)
                print(f"🛠️ Tool Result: {tool_result}")
                # 提取工具执行结果（ToolMessage）
                response.append(tool_result["messages"][0])
                # 使用语法高亮显示工具结果
                self.console.print(
                    Panel.fit(
                        Syntax(
                            "\n" + tool_result["messages"][0].content + "\n", "text"
                        ),
                        title="Tool Result",
                    )
                )
            except Exception as e:
                # 工具执行失败时，创建错误消息
                # 必须保留 tool_call_id，以便模型能够正确关联错误和原始请求
                response.append(
                    ToolMessage(
                        content=f"ERROR: Exception during tool '{tool_name}' execution: {e}",
                        tool_call_id=tc["id"],
                    )
                )
                # 使用红色边框显示错误信息
                self.console.print(
                    Panel.fit(
                        Markdown(
                            f"**ERROR**: Exception during tool '{tool_name}' execution: {e}"
                        ),
                        title="Tool Error",
                        border_style="red",
                    )
                )
            # else:
            #     # Handle rejection or modification
            #     pass
        
        # 返回所有工具执行的结果消息
        return {"messages": response}

    def print_mermaid_workflow(self):
        """
        工具方法：打印工作流图的可视化
        
        尝试生成 Mermaid 格式的 PNG 图片，如果失败则：
        1. 尝试生成 Mermaid 文本格式并在控制台显示
        2. 生成 ASCII 格式的图表
        
        输出文件：langgraph_workflow.png（如果成功）
        """
        try:
            # 尝试生成 PNG 格式的工作流图
            mermaid = self.agent.get_graph().draw_mermaid_png(
                output_file_path="langgraph_workflow.png",
                max_retries=5,  # 最多重试 5 次
                retry_delay=2,  # 每次重试延迟 2 秒
            )
        except Exception as e:
            # PNG 生成失败时，回退到文本格式
            print(f"Error generating mermaid PNG: {e}")
            # 生成 Mermaid 文本格式
            mermaid = self.agent.get_graph().draw_mermaid()
            # 在控制台中使用语法高亮显示 Mermaid 代码
            self.console.print(
                Panel.fit(
                    Syntax(mermaid, "mermaid", theme="monokai", line_numbers=False),
                    title="Workflow (Mermaid)",
                    border_style="cyan",
                )
            )
            # 打印 ASCII 格式的工作流图（更简洁的文本表示）
            print(self.agent.get_graph().draw_ascii())
