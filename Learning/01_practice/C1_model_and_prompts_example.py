import os
import asyncio
from typing import Optional, List

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import Field, BaseModel

# 加载环境变量
load_dotenv()

# 结构化输出定义
class UserProfile(BaseModel):
    """用户信息结构化输出模型 """
    name: str = Field(...,description="用户的姓名" , min_length=2, max_length=5)
    age: Optional[int] = Field(None,description="用户的年龄，如果未提及则为空")
    interest: List[str] = Field(default_factory=list,description="用户的兴趣爱好")

async def main():

    #1.llm模块入口：
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        model=os.getenv("MODEL_NAME"),
        temperature=0.7,
    )

    # #Message与同步调用（invoke）
    # messages = [
    #     SystemMessage(content= "你是一个专业的文本信息提取助手"),
    #     HumanMessage(content ="我叫张三，今年28岁，喜欢打篮球和看科幻小说。")
    # ]
    # print("1.测试基础Invoke")
    # response = await llm.ainvoke(messages)
    # print(f"->原始AIMessage content：{response.content}")

    #2. Bind 绑定运行期参数
    bound_llm = llm.bind(temperature=0.1)

    #PromptTemplate 与 MessagesPlaceholder
    prompt = ChatPromptTemplate.from_messages([
        ("system","你是一个名叫{bot_name}的助手，请以JSON格式回答"),
        #MessagesPlaceholder  用于动态插入多轮聊天记录
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{user_input}")
    ])

    print("\n 测试结构化输出（）(模拟 Agent 底层 bind_tools 逻辑):")

    #create_react_agent 底层所做
    llm_with_tools = llm.bind_tools([UserProfile])

    chain_structured = prompt | llm_with_tools

    ai_msg = await chain_structured.ainvoke({
        "bot_name":"信息收集者",
        "chat_history":[],
        "user_input":"我叫李华，平时喜欢听音乐和跑步。"
    })




if __name__ == "__main__":
    asyncio.run(main())