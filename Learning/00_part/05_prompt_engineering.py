"""
0.2 Prompt 工程技巧：生产环境最佳实践 (Pydantic 结构化输出、模板管理、Partial Formatting)
"""
import logging
from datetime import datetime
from typing import List
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_API_URL"),
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=0.1 # 抽取任务通常需要较低的温度以保证确定性
)

# 1. 生产环境的结构化输出 (Structured Output) 强约束
# 生产环境中很少要求模型"输出 JSON 字符串"，而是直接利用 with_structured_output 和 Pydantic 模型
class EntityExtractionResult(BaseModel):
    """实体抽取结果"""
    persons: List[str] = Field(description="文本中出现的所有人名列表", default_factory=list)
    locations: List[str] = Field(description="文本中出现的所有地名列表", default_factory=list)
    organization: List[str] = Field(description="文本中出现的所有公司、机构名称", default_factory=list)

def extract_entities_production(text: str) -> EntityExtractionResult | None:
    """生产级的实体抽取：严格校验输出格式"""
    logger.info(f"开始提取实体，输入文本: {text[:15]}...")
    
    # with_structured_output 会强制 LLM 返回符合 Pydantic 定义的对象
    structured_llm = llm.with_structured_output(EntityExtractionResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的 NLP 实体抽取引擎。不要返回任何多余的解释文字。"),
        ("human", "请提取以下文本中的实体:\n\n{input_text}")
    ])
    
    chain = prompt | structured_llm
    
    try:
        result = chain.invoke({"input_text": text})
        return result
    except Exception as e:
        logger.error(f"实体抽取失败: {e}")
        return None

# 2. 生产环境提示词管理：动态上下文与 Partial Variables
def create_customer_service_prompt() -> ChatPromptTemplate:
    """创建一个带有当前系统时间的智能客服 Prompt 模板"""
    
    # 在生产环境中，诸如当前时间、日期等变量通常是"部分格式化 (partial)"注入的
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个高端品牌的售后助理。\n当前系统时间是：{current_time}\n公司政策：不支持无理由退货。"),
        ("human", "用户问题：{query}")
    ])
    
    # 动态绑定当前时间，这样每次调用时只需要传入 query
    def get_current_time():
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    return prompt.partial(current_time=get_current_time)

if __name__ == "__main__":
    print("--- 1. 生产级结构化输出 (Pydantic 约束) ---")
    text = "昨天，张三和李四代表腾讯公司去了北京的总部开会，讨论下个季度的计划。"
    entities = extract_entities_production(text)
    if entities:
        print(f"解析成功:\n- 人物: {entities.persons}\n- 地点: {entities.locations}\n- 机构: {entities.organization}")
        
    print("\n--- 2. 生产级 Prompt 模板管理 (注入动态上下文) ---")
    cs_prompt = create_customer_service_prompt()
    # 运行时只需要关注业务参数，时间已被 partial 函数自动处理
    messages = cs_prompt.invoke({"query": "我现在申请退货可以吗？"}).to_messages()
    
    print("最终生成的 Messages:")
    for msg in messages:
        print(f"[{msg.type.upper()}]:\n{msg.content}\n")
