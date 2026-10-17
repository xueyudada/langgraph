import asyncio
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

"""
LangChain 中 ChatPromptTemplate 的核心与常用方法全景解析。
ChatPromptTemplate 用于将动态变量组装成发送给大模型的对话格式 (Messages)。
"""

async def main():
    print("==========================================")
    print(" 1. 初始化方法 (Class Methods / 创建模板)")
    print("==========================================")

    # 1.1 from_messages(): 最强大、最常用的初始化方法
    # 可以混用元组 ("角色", "内容")、具体 Message 类、Placeholder
    print("\n[from_messages] 混合初始化...")
    prompt_from_msg = ChatPromptTemplate.from_messages([
        ("system", "你是一个名叫 {bot_name} 的助手。"),
        MessagesPlaceholder(variable_name="history"), # 用于动态插入多轮聊天记录
        ("human", "{user_input}")
    ])
    print(f"创建的模板包含 {len(prompt_from_msg.messages)} 个消息片段")

    # 1.2 from_template(): 最简捷的方法，如果你只需要一轮对话（底层会将其转为单条 HumanMessage）
    print("\n[from_template] 单行字符串初始化...")
    prompt_single = ChatPromptTemplate.from_template("请将这段话翻译成 {language}：{text}")
    
    # 1.3 from_role_strings() (较少用，类似 from_messages，纯字符串形式)

    print("\n==========================================")
    print(" 2. 运行时格式化方法 (Formatting)")
    print("==========================================")
    
    # 准备假数据
    variables = {
        "bot_name": "贾维斯",
        "history": [HumanMessage(content="你好"), AIMessage(content="你好！")],
        "user_input": "今天天气怎么样？"
    }

    # 2.1 format_messages(): 将变量填入模板，直接返回一个 List[BaseMessage]
    # 这是最底层、也最能看清数据结构的格式化方法
    print("\n[format_messages] 格式化为消息列表...")
    formatted_msgs = prompt_from_msg.format_messages(**variables)
    for msg in formatted_msgs:
        print(f" - {msg.__class__.__name__}: {msg.content}")

    # 2.2 format(): 将所有消息拼成一整个长字符串返回
    # 通常不推荐在 Chat 模型上使用，但在需要打印日志或兼容旧版 LLM 时有用
    print("\n[format] 格式化为纯文本字符串...")
    formatted_str = prompt_from_msg.format(**variables)
    print(repr(formatted_str))

    # 2.3 format_prompt(): 格式化为 PromptValue 对象
    # PromptValue 是一个中间态包装器，可以调用 .to_string() 或 .to_messages()
    print("\n[format_prompt] 格式化为 PromptValue包装器...")
    prompt_value = prompt_from_msg.format_prompt(**variables)
    # print(prompt_value.to_messages()) 


    print("\n==========================================")
    print(" 3. 继承自 Runnable 的核心机制 (LCEL)")
    print("==========================================")

    # 3.1 invoke() / ainvoke(): 实际上在 LCEL 链中，它就是调用了 format_prompt
    print("\n[invoke] 标准的 LCEL 调用...")
    result = prompt_from_msg.invoke(variables)
    print(f"Invoke 返回类型: {type(result).__name__}") # PromptValue
    
    # 3.2 partial(): 偏函数绑定变量。非常重要！
    # 比如 bot_name 在一开始就固定了，但 user_input 每次都变，我们可以提前填入 bot_name
    print("\n[partial] 提前绑定部分变量...")
    partial_prompt = prompt_from_msg.partial(bot_name="奥创")
    
    # 此时再次调用，只需要传入剩下的历史和输入即可
    final_result = partial_prompt.invoke({
        "history": [], 
        "user_input": "你是谁？"
    })
    print(f"偏函数渲染后: {final_result.to_messages()[0].content}")
    
    # 3.3 | (管道符): 作为 Runnable，随时可以和模型组装成链
    print("\n[ | 操作符] 组装链条...")
    # chain = prompt_from_msg | ChatOpenAI()
    print("prompt | llm -> RunnableSequence 构建完成")

if __name__ == "__main__":
    asyncio.run(main())
