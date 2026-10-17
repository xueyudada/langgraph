"""
0.1 Python + 工程基础
- 装饰器 (Decorators)
- 类型注解 (Typing)
- JSON 操作
- 工程基础概念 (uv, venv, Git, Docker, REST API)
"""
import json
import time
from typing import List, Dict, Optional, Any, Callable

# 1. 装饰器 (Decorator) - 模拟 LangChain @tool
class Tool:
    """模拟 LangChain 中的 Tool 类，封装大模型可调用的工具"""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, *args: Any, **kwargs: Any) -> Any:
        print(f"[Tool Execution] 正在执行工具 '{self.name}'...")
        return self.func(*args, **kwargs)

def tool(func: Callable) -> Tool:
    """
    模拟 LangChain 的 @tool 装饰器核心逻辑。
    它提取函数的名称(name)和文档字符串(docstring作为description)，
    将普通函数变成大模型可以理解的 Tool 对象。
    """
    name = func.__name__
    description = func.__doc__ or "No description"
    return Tool(name=name, description=description.strip(), func=func)

@tool
def get_weather(location: str) -> str:
    """获取指定城市的当前天气。"""
    # 真实场景中这里会发送网络请求调用天气 API
    return f"{location} 晴天，25度。"

# 2. 类型注解 (Typing) & JSON
def parse_api_response(json_str: str) -> Optional[Dict[str, Any]]:
    """解析 JSON 字符串，并使用类型注解明确返回类型"""
    try:
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError:
        print("JSON 解析失败")
        return None

if __name__ == "__main__":
    print("--- 装饰器 (@tool) 演示 ---")
    # 此时 get_weather 已从普通函数变成了 Tool 对象
    print(f"工具名提取: {get_weather.name}")
    print(f"工具描述提取: {get_weather.description}")
    
    # 模拟大模型生成的调用
    result = get_weather.run(location="北京")
    print(f"返回给模型的结果: {result}\n")

    print("--- JSON & 类型注解演示 ---")
    mock_json = '{"user": "Alice", "age": 25, "is_active": true}'
    parsed = parse_api_response(mock_json)
    print(f"解析后的字典: {parsed}")

    """
    【工程基础概念笔记】
    - 虚拟环境 (uv/venv): 用于隔离项目的 Python 依赖。推荐使用 `uv venv` 和 `uv pip install`，速度极快。
    - Git: 版本控制系统，常用命令 `git init`, `git add`, `git commit`, `git push`。
    - Docker: 容器化技术，通过 Dockerfile 将应用及其环境打包运行。
    - REST API: 表述性状态转移，基于 HTTP 协议的网络接口标准（GET, POST, PUT, DELETE）。
    """
