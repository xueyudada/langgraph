"""
0.2 高级 API 功能：JSON Mode (结构化输出) 与 Function Calling (工具调用)
这两个特性是将 LLM 接入业务系统（如 Agent 开发）的核心。
"""
from pydantic import BaseModel

# 1. 结构化输出 (JSON Mode)
class WeatherResponse(BaseModel):
    location: str
    temperature: float
    description: str

def demo_json_mode() -> None:
    """
    JSON Mode 要求大模型返回合法的 JSON 格式。
    在最新的 OpenAI API 中，可以使用 response_format 参数结合 Pydantic 模型强制输出结构。
    """
    print("--- JSON Mode / 结构化输出 ---")
    print("调用示例：")
    print("""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "提取天气信息。"},
            {"role": "user", "content": "今天上海气温 25度，天气晴朗。"}
        ],
        response_format=WeatherResponse, # 直接传入 Pydantic 模型
    )
    weather_info = completion.choices[0].message.parsed
    print(weather_info.temperature) # 25.0
    """)

# 2. Function Calling
def demo_function_calling() -> None:
    """
    Function Calling 允许模型选择调用预定义的函数，而不是直接回复文本。
    常用于让模型查询数据库、调用外部 API 获取实时信息。
    """
    print("\n--- Function Calling / 工具调用 ---")
    tools_definition = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "获取指定城市的当前天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "城市名，例如：北京, 上海"
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    print("定义给 LLM 的工具列表 (tools 参数)：")
    print(tools_definition)
    print("""
    # API 调用时传入 tools 参数
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
        tools=tools_definition
    )
    
    # 检查模型是否决定调用函数
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        print(f"模型想要调用的函数名: {tool_call.function.name}")
        print(f"传入的参数: {tool_call.function.arguments}")
        
        # 接下来：在本地执行该函数，然后将结果再回传给大模型
    """)

if __name__ == "__main__":
    demo_json_mode()
    demo_function_calling()
