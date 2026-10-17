"""
0.2 大模型 API 调用：生产环境最佳实践 (重试、异常处理、流式输出、工具调用)
"""
import asyncio
import logging
import os
from typing import AsyncGenerator, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from openai import APIError

# 配置生产环境标准的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 初始化生产级别的 LLM 客户端
# 生产环境通常需要配置: max_retries, timeout, 甚至不同的回退模型 (Fallbacks)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.7,
    max_tokens=1000,
    max_retries=3,          # 生产环境必备：网络波动时的自动重试机制
    timeout=60.0            # 生产环境必备：防止请求无限期挂起
)

async def robust_chat_completion(messages: list) -> str | None:
    """带有完善异常处理的基础对话调用"""
    try:
        logger.info(f"发送请求给 LLM，消息条数: {len(messages)}")
        response = await llm.ainvoke(messages)
        
        # 记录 Token 使用量，用于生产环境的成本核算和监控
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            logger.info(f"Token 消耗: {response.usage_metadata}")
            
        return response.content
        
    except APIError as e:
        logger.error(f"调用 LLM API 发生网络或服务错误: {str(e)}")
    except Exception as e:
        logger.exception("发生未预期的系统异常")
    
    return None

async def stream_response(messages: list) -> AsyncGenerator[str, None]:
    """生产环境中的流式输出，通常会通过 WebSocket 推送给前端"""
    try:
        logger.info("开始获取流式响应...")
        async for chunk in llm.astream(messages):
            # 在实际业务中，这里通常是 yield chunk 并发送到消息队列或 WebSocket
            yield chunk.content
    except asyncio.TimeoutError:
        logger.error("LLM 流式响应超时")
        yield "\n[Error: 请求超时，请稍后再试]"
    except Exception as e:
        logger.exception("流式输出时发生异常")
        yield f"\n[Error: 服务内部错误]"

@tool
def get_order_status(order_id: str) -> dict[str, Any]:
    """查询电商系统中的订单状态。
    
    参数:
        order_id: 订单的唯一标识符，通常是 'ORD-' 开头的字符串。
    """
    # 模拟查询数据库
    logger.info(f"正在查询数据库获取订单 {order_id} 的状态...")
    if order_id == "ORD-12345":
        return {"status": "已发货", "carrier": "SF Express", "tracking_no": "SF111222"}
    return {"status": "未找到订单", "error": "Order not found"}

async def execute_tool_calling() -> None:
    """生产级别的工具调用链路"""
    llm_with_tools = llm.bind_tools([get_order_status])
    messages = [HumanMessage(content="帮我查一下订单 ORD-12345 的物流状态。")]
    
    try:
        response = await llm_with_tools.ainvoke(messages)
        if response.tool_calls:
            logger.info(f"LLM 决定调用工具: {response.tool_calls}")
            # 生产环境中，此处会解析 tool_calls，真实执行对应的 Python 函数并返回结果给 LLM
            # 为演示简便，这里仅做拦截和打印
        else:
            logger.info(f"LLM 拒绝或无需调用工具，直接回复: {response.content}")
    except Exception as e:
        logger.exception("工具调用执行链路失败")

async def main():
    print("\n--- 1. 鲁棒性基础对话 ---")
    msgs = [SystemMessage(content="你是一个专业的客服助理。"), HumanMessage(content="你们支持退货吗？")]
    reply = await robust_chat_completion(msgs)
    print("回复:", reply)
    
    print("\n--- 2. 流式输出处理 ---")
    stream_msgs = [HumanMessage(content="写一句关于奋斗的简短名言。")]
    async for text_chunk in stream_response(stream_msgs):
        print(text_chunk, end="", flush=True)
    print("\n")
    
    print("\n--- 3. 工具调用模拟 ---")
    await execute_tool_calling()

if __name__ == "__main__":
    asyncio.run(main())
