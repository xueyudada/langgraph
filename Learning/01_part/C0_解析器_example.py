import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 加载环境变量
load_dotenv()

# ==========================================================
# 1. 定义 Pydantic 模型 (替代官方示例中手写的 JSON 字符串)
# ==========================================================
class ExamParse(BaseModel):
    """解析考试题目和答案"""
    question: str = Field(description="考试文本中的问题")
    answer: str = Field(description="考试文本中给出的答案")


def main():
    print("=== 最稳定实践：PydanticParser + JSON Mode ===\n")
    
    # 2. 构造解析器，它会基于上面的 Pydantic 类，自动生成完美的 JSON 格式指令！
    parser = PydanticOutputParser(pydantic_object=ExamParse)
    format_instructions = parser.get_format_instructions()
    
    # 你可以打印出来看看，LangChain 帮你写的格式指令有多详细
    print("-> LangChain 自动生成的格式指令 (Format Instructions):\n")
    print(format_instructions)
    print("\n" + "="*50 + "\n")

    # 3. 构造 Prompt，把格式指令塞进系统提示词，就像 DeepSeek 官方做的那样
    # 注意：官方强调 prompt 中必须包含 "json" 字眼，format_instructions 里天然自带了！
    prompt = PromptTemplate(
        template="""The user will provide some exam text. Please parse the "question" and "answer".
        
{format_instructions}

USER TEXT:
{text}
""",
        input_variables=["text"],
        partial_variables={"format_instructions": format_instructions},
    )

    # 4. 初始化 LLM，严格按照官方要求加上 JSON Mode 开关
    llm = ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        model=os.getenv("MODEL_NAME", "deepseek-chat"),
        temperature=0.1, # 提取信息时，温度尽量调低
        # 这里是核心！开启大模型的强约束
        model_kwargs={"response_format": {"type": "json_object"}} 
    )

    # 5. 组装 LCEL 链条
    # 流程：组装带有严格格式要求的 Prompt -> 发给处于 JSON 模式的大模型 -> 用解析器把返回的 JSON 强转为 Pydantic 对象
    chain = prompt | llm | parser

    # 6. 直接调用！
    user_input = "Which is the longest river in the world? The Nile River."
    print(f"-> 发送测试文本: {user_input}\n")
    
    result = chain.invoke({"text": user_input})
    
    print(f"-> 最终拿到的原汁原味 Python 对象: {type(result)}")
    print(f"-> Question: {result.question}")
    print(f"-> Answer: {result.answer}")


if __name__ == "__main__":
    main()
