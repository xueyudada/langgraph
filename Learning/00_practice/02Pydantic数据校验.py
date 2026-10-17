from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError


class UserProfile(BaseModel):
    """定义一个用户画像数据模型，带有严格的类型和约束"""
    username: str = Field(...,description = "用户名",min_length=3,max_length=20)
    age: int = Field(...,description="用户年龄",ge=0,le=20)
    email: Optional[str] = Field(None,description="用户邮箱")
    tags: List[str] = Field(default_factory=list,description="用户兴趣标签")

def process_user_data(data: dict)-> None:
    try:
        # Pydantic 自动验证传入的字典并转换为对象
        user = UserProfile(**data)
        print("数据校验成功")
        print(f"解析后的对象：{user}")
        print(f"转回 JSON ：{user:model_dump_json(indent=2)}")
        #将user转化为json格式，indent缩进格式
    except ValidationError as e:
        print("数据校验失败")
        print(e.json(indent=2))


if __name__ == '__main__':
    print("---测试正确数据---")
    valid_date = {
        "username": "LangGraph_Dev",
        "age": 18,
        "email": "dev@example.com",
        "tags": ["python", "javascript"]
    }
    process_user_data(valid_date)

    print("\n--- 测试错误数据 (年龄超出范围，用户名太短) ---")
    invalid_data = {
        "username": "ab",
        "age": 150,
        "tags": "这不是一个列表"  # 类型错误
    }
    process_user_data(invalid_data)