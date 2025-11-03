"""
主程序入口文件
CLI 程序的启动点，负责初始化 Agent 并启动交互循环
"""
from agent import Agent
import asyncio


async def async_main():
    """
    异步主函数
    创建 Agent 实例，初始化资源，启动工作流，并在退出时清理资源
    """
    # 创建 Agent 实例
    agent = Agent()
    
    # 异步初始化：加载工具、配置 MCP 客户端、编译工作流图
    await agent.initialize()
    
    # 打印工作流可视化图（Mermaid 格式）
    agent.print_mermaid_workflow()
    
    # 启动主循环：触发工作流的第一次执行
    # 注意：实际的用户交互循环是在 LangGraph 工作流图中实现的
    # 工作流节点之间的边形成了循环：user_input -> model_response -> (tool_use or user_input)
    await agent.run()
    
    # 清理资源：关闭数据库检查点连接
    await agent.close_checkpointer()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(async_main())
