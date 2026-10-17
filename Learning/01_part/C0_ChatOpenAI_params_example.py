import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

"""
LangChain 中 ChatOpenAI 初始化参数全景解析示例
这些参数用于精细化控制 OpenAI (或兼容的 API) 的行为。
"""

# ==========================================
# 1. 核心模型与鉴权参数 (Core & Auth)
# ==========================================
api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
organization = os.getenv("OPENAI_ORGANIZATION", "org-...")
project = os.getenv("OPENAI_PROJECT", "proj-...")

# ==========================================
# 2. 基础生成控制参数 (Generation Control)
# ==========================================
# temperature: 采样温度，0.0 到 2.0。值越高输出越随机(如0.8)，值越低输出越稳定/确定(如0.2)。
# max_tokens: 限制模型单次生成的最大 token 数量。如果不设置，默认为模型允许的最大上限。
# top_p: 核采样，0.0 到 1.0。例如 0.1 意味着只考虑前 10% 概率的 token。通常不与 temperature 同时更改。
# stop: 停止词列表，当模型生成这些词时会立即停止生成。

# ==========================================
# 3. 惩罚与偏差参数 (Penalties & Bias)
# ==========================================
# presence_penalty: 存在惩罚，-2.0 到 2.0。正值根据新 token 是否已经出现在文本中来惩罚它们，增加模型谈论新话题的可能性。
# frequency_penalty: 频率惩罚，-2.0 到 2.0。正值根据新 token 在文本中已有的频率来惩罚它们，降低模型逐字重复的可能性。
# logit_bias: 对特定 token 的生成概率进行人为修改。字典格式，key是token ID(字符串)，value是-100到100的偏差值。

# ==========================================
# 4. 网络请求与执行参数 (Network & Execution)
# ==========================================
# timeout: 请求超时时间（秒）。可以是浮点数，也可以是 (connect_timeout, read_timeout) 元组。
# max_retries: 请求失败时的最大自动重试次数。
# streaming: 是否开启流式输出。设为 True 后可以通过 .stream() 逐步获取生成内容。

# ==========================================
# 5. 其他高级参数 (Advanced)
# ==========================================
# seed: 随机种子，设置后模型会尽力保证每次输出的确定性（但不是绝对保证）。
# model_kwargs: 用于传递那些 LangChain 尚未原生支持，但 OpenAI API 已经支持的额外参数（如 response_format 等）。
# default_headers: 每次请求附带的自定义请求头。
# default_query: 每次请求附带的自定义 URL 查询参数。


def create_comprehensive_llm():
    """创建一个包含了各种参数配置的 ChatOpenAI 实例"""
    
    llm = ChatOpenAI(
        # --- 核心与鉴权 ---
        model="gpt-4o",                    # 模型名称 (如 gpt-4o, gpt-3.5-turbo 等)
        api_key=api_key,                   # API Key (通常直接读环境变量 OPENAI_API_KEY)
        base_url=base_url,                 # API 基础URL (常用于代理转发)
        organization=organization,         # 组织 ID (可选)
        # project=project,                 # 项目 ID (可选，部分新版 SDK 支持)
        
        # --- 生成控制 ---
        temperature=0.7,                   # 创造力温度
        max_tokens=1024,                   # 最大生成 token 数
        top_p=1.0,                         # 替代 temperature 的采样方法
        stop=["</think>", "User:"],        # 停止序列
        
        # --- 惩罚与偏差 ---
        presence_penalty=0.1,              # 鼓励模型聊新话题
        frequency_penalty=0.1,             # 抑制模型重复用词
        logit_bias={"50256": -100},        # 强烈抑制特定 token (这里随便举例的ID)
        
        # --- 网络与执行 ---
        timeout=60.0,                      # 超时时间 60秒
        max_retries=3,                     # 失败重试 3次
        streaming=True,                    # 开启流式响应
        
        # --- 高级参数 ---
        seed=42,                           # 固定随机种子
        model_kwargs={                     
            # 传递额外的 OpenAI 原生参数
            "user": "user_12345",          # 传递终端用户ID，用于OpenAI滥用监控
            # "response_format": {"type": "json_object"} # 强制输出 JSON 格式 (需要系统提示词配合)
        },
        default_headers={"X-Custom-Header": "LangChain-App"}, 
        default_query={"api_version": "2024-02-15-preview"}   # (主要用于 Azure 等兼容平台)
    )
    
    return llm

if __name__ == "__main__":
    print("正在初始化包含全面参数的 ChatOpenAI 实例...")
    llm = create_comprehensive_llm()
    
    print("\n创建成功！当前 LLM 的一些关键配置如下：")
    print(f"- 模型名称 (model): {llm.model_name}")
    print(f"- 采样温度 (temperature): {llm.temperature}")
    print(f"- 最大生成长度 (max_tokens): {llm.max_tokens}")
    print(f"- 失败重试次数 (max_retries): {llm.max_retries}")
    
    print("\n正在尝试流式调用该模型...")
    try:
        # 注意：这里需要你有真实的 API Key 才能成功请求
        response_stream = llm.stream([HumanMessage(content="用一句话解释什么是大语言模型。")])
        print("模型响应: ", end="", flush=True)
        for chunk in response_stream:
            print(chunk.content, end="", flush=True)
        print()
    except Exception as e:
        print(f"\n[提示] 调用失败，可能是未配置正确的 API Key 或网络原因。错误信息:\n{e}")
