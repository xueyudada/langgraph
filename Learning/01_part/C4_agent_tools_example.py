import os
import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# 【模块 5】LangGraph 相关入口
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")

# 1. 定义工具 (@tool)
# 必须包含清晰的 docstring 和类型注解，模型依赖这些信息生成 tool_call
@tool
def get_weather(location: str) -> str:
    """获取指定城市的天气状况。如果用户问天气，必须调用此工具。"""
    # 模拟真实世界 API 响应
    return f"{location} 现在的天气是多云，气温 22°C。"

@tool
def calculate_expression(expression: str) -> float:
    """计算数学表达式的值，参数 expression 必须是合法的数学字符串，如 '25 * 4'"""
    try:
        # 注意: 生产环境中禁止直接使用 eval，这里仅作演示
        return float(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

async def main():
    print("=== 模块 5: Tools 与 Agent (基于 LangGraph) ===\n")
    
    llm = ChatOpenAI(
        api_key=API_KEY, 
        base_url=BASE_URL, 
        model=MODEL_NAME, 
        temperature=0
    )
    
    # 2. 将工具绑定到模型
    tools = [get_weather, calculate_expression]
    # 如果仅测试模型的 Function Calling 能力而不做循环，可使用 bind_tools
    # bound_llm = llm.bind_tools(tools) 
    
    # 3. 记忆 / 持久化 (LangGraph Checkpointer)
    # 取代了 LangChain 旧版的 Memory 类
    memory = MemorySaver()
    
    # 4. 创建标准 Agent (取代旧版的 AgentExecutor)
    # create_react_agent 是底层由 LangGraph 图构建的高级封装
    agent = create_react_agent(llm, tools, checkpointer=memory)
    
    # 配置 thread_id 用于区分不同用户的上下文记忆
    config = {"configurable": {"thread_id": "user_777"}}
    
    print("[1] 第一轮对话 (测试 Agent 的工具调度):")
    res1 = await agent.ainvoke(
        {"messages": [HumanMessage(content="北京天气怎样？顺便帮我算下 1024 除以 8 的结果。")]},
        config=config
    )
    
    # 遍历对话历史打印执行过程
    for msg in res1["messages"]:
        if msg.type == "ai" and msg.content:
            print(f" -> [AI 回复]: {msg.content}")
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            print(f" -> [Agent 发起工具调用]: {msg.tool_calls}")
        elif msg.type == "tool":
            print(f" -> [工具返回]: {msg.name} = {msg.content}")
            
    print("\n[2] 第二轮对话 (测试 Checkpointer 的记忆能力):")
    res2 = await agent.ainvoke(
        {"messages": [HumanMessage(content="根据我刚才查天气的那个城市，那里适合穿外套吗？")]},
        config=config
    )
    
    # 打印最终答复
    print(f"\n-> [AI 最终答复]: {res2['messages'][-1].content}")
    
    # (备注: 模块 7 MCP 的工具接入，相当于将外部的 MCP 工具拼接到这里的 tools 列表中，详见 B1 示例)

if __name__ == "__main__":
    asyncio.run(main())
