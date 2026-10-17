import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, AliasChoices, ValidationError


def generate_default_tags():
    """用于 default_factory 的工厂函数，避免可变类型的共享内存问题"""
    return ["new", "unverified"]


class UserProfile(BaseModel):
    # ==========================================
    # 1. 默认值与必填控制 (Defaults)
    # ==========================================
    
    # 必填字段 (使用 ...)
    username: str = Field(..., description="用户的唯一登录名")
    
    # 非必填字段，指定具体默认值
    status: str = Field(default="active", description="用户账号状态")
    
    # 使用 default_factory 动态生成默认值 (避免所有实例共享同一个列表或UUID)
    user_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="自动生成的唯一ID")
    tags: List[str] = Field(default_factory=generate_default_tags)


    # ==========================================
    # 2. 字段别名与解析控制 (Aliases & Parsing)
    # ==========================================
    
    # 统一别名：输入输出都用 'camelCaseName' 替代 'full_name'
    full_name: str = Field(default="Unknown", alias="fullName")
    
    # 验证别名：读取(解析)时，可以接受 'emailAddress' 或 'email_address'
    email: str = Field(
        ..., 
        validation_alias=AliasChoices('emailAddress', 'email_address')
    )
    
    # 序列化别名：在转为 JSON 字典输出时，该字段会被重命名为 'X-API-KEY'
    api_key: Optional[str] = Field(None, serialization_alias="X-API-KEY")
    
    # 严格模式：拒绝隐式类型转换 (比如传入 "25" 会报错，必须是数字 25)
    login_count: int = Field(default=0, strict=True)


    # ==========================================
    # 3. 数据约束验证 (Validation Constraints)
    # ==========================================
    
    # 数字类约束：年龄必须 >= 18 且 <= 120，且必须是 1 的倍数(整数)
    age: int = Field(..., ge=18, le=120, multiple_of=1)
    
    # 序列类约束：密码长度在 8 到 20 之间
    password_hash: str = Field(..., min_length=8, max_length=20)
    
    # 正则表达式验证：简单的手机号正则验证
    phone_number: str = Field(..., pattern=r'^1[3-9]\d{9}$')


    # ==========================================
    # 4. 输出与打印控制 (Serialization & Repr)
    # ==========================================
    
    # exclude: 转为字典/JSON时，密码和内部备注不会被输出 (保护敏感信息)
    internal_notes: str = Field(default="No notes", exclude=True)
    
    # repr: 在 print(user_obj) 打印终端时隐藏，保持清爽，但 dumped 时依然在
    heavy_data_payload: str = Field(default="very_long_string...", repr=False)
    
    # frozen: 创建后就变成只读属性，无法再被修改
    created_at: datetime = Field(default_factory=datetime.now, frozen=True)


    # ==========================================
    # 5. 文档与元数据 (Metadata & JSON Schema)
    # ==========================================
    
    # 这些主要服务于 OpenAPI 文档生成 或 LLM 的 Prompt 提示
    role: str = Field(
        default="user",
        title="用户角色",
        description="定义用户的系统权限级别，如 admin, user, guest",
        examples=["admin", "guest"],
        deprecated=False,
        json_schema_extra={"ui_component": "dropdown_menu"} # 注入自定义信息到Schema
    )

# --- 示例使用 ---

if __name__ == "__main__":
    # 1. 成功实例化的例子 (注意验证别名 emailAddress 的使用)
    user_data = {
        "username": "coder_jack",
        "emailAddress": "jack@example.com",  # 命中 validation_alias
        "fullName": "Jack Smith",            # 命中 alias
        "age": 25,
        "password_hash": "secret_abc",
        "phone_number": "13812345678",
        "login_count": 5
    }
    
    user = UserProfile(**user_data)
    
    print("--- 1. 实例化对象打印 (注意 heavy_data_payload 不在打印结果中) ---")
    print(user)
    
    print("\n--- 2. 导出为字典 (注意 internal_notes 被排除了，api_key 变成了 X-API-KEY) ---")
    print(user.model_dump(by_alias=True)) # by_alias=True 激活序列化别名
    
    print("\n--- 3. 尝试修改 frozen (只读) 字段会报错 ---")
    try:
        user.created_at = datetime.now()
    except ValidationError as e:
        print(f"验证错误: {e.errors()[0]['msg']}")
        
    print("\n--- 4. strict 严格模式测试 ---")
    try:
        # 传入字符串 "10" 给严格要求 int 的 login_count 会报错
        invalid_user = UserProfile(**{**user_data, "login_count": "10"})
    except ValidationError as e:
        print(f"验证错误: {e.errors()[0]['msg']}")

    print("\n--- 5. 约束测试 (年龄 < 18) ---")
    try:
        invalid_user = UserProfile(**{**user_data, "age": 16})
    except ValidationError as e:
        print(f"验证错误: {e.errors()[0]['msg']}")
        
    print("\n--- 6. 生成的 JSON Schema (包含描述、示例和自定义信息) ---")
    # 这部分常常被 FastAPI 用于生成 Swagger UI 文档或给大模型看
    schema = UserProfile.model_json_schema()
    print("角色字段(role)的Schema片段:")
    print(schema['properties']['role'])
