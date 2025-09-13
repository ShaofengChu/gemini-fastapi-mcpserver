import json
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from jose import JWTError, jwt
from passlib.context import CryptContext

# 加载 .env 文件中的环境变量
load_dotenv()

# --- 认证相关配置 ---
# 建议将 SECRET_KEY 存储在环境变量中，并且是一个复杂的随机字符串
# 你可以在 shell 中用 `openssl rand -hex 32` 生成一个
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise ValueError("请设置 SECRET_KEY 环境变量用于 JWT 签名")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 用于密码哈希
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2PasswordBearer 会在 /token (我们即将创建的) URL 上寻找 token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 模拟用户数据库 ---
# 在真实应用中，这里应该是数据库查询
fake_users_db = {
    "testuser": {
        "username": "testuser",
        "full_name": "Test User",
        "email": "test@example.com",
        # 密码应该是哈希过的，"password" 的哈希值是 "$2b$12$EixZA0C5T.VwORSo5g4NAeQuG2V.TjDaO/j2O.aLaaP2Yc/I2pYfS"
        "hashed_password": pwd_context.hash("password"),
        "disabled": False,
    }
}

# 从环境变量中获取 Google API Key
# 建议在部署时使用环境变量，而不是硬编码
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("请设置 GOOGLE_API_KEY 环境变量")

# 初始化 Google Gemini 客户端
genai.configure(api_key=api_key)

app = FastAPI()

# --- Pydantic 数据模型 ---
class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None

class Token(BaseModel):
    access_token: str
    token_type: str


# 定义请求数据模型，用于接收客户端的指令
class CommandRequest(BaseModel):
    user_command: str

# 核心工具函数（你的工具库）
# 示例：一个获取天气的工具
def get_current_weather(location: str):
    """
    获取指定地点的当前天气

    Args:
        location: 城市名，例如 "上海", "东京"
    """
    print(f"正在调用天气工具，查询 {location} 的天气...")
    # 实际项目中，这里会调用一个外部天气 API
    # 为了快速演示，我们返回一个假数据
    if location == "上海":
        return {"location": "上海", "temperature": "25", "unit": "摄氏度", "description": "晴朗"}
    else:
        return {"location": location, "temperature": "不详", "unit": "", "description": "未知"}

# 初始化 Gemini 模型，并告诉它我们有哪些可用的工具
# Gemini SDK 可以直接从 Python 函数定义中推断出工具的结构
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-latest", # 使用最新的 Flash 模型，它速度快且广泛可用
    tools=[get_current_weather]
)

# --- 认证辅助函数 ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return User(**user_dict)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_active_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

# --- 路由定义 ---

# 登录路由，用于获取 JWT Token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = get_user(fake_users_db, form_data.username)
    if not user or not verify_password(form_data.password, fake_users_db[user.username]["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# 路由定义：处理客户端的 POST 请求
@app.post("/command")
async def process_command(request: CommandRequest, current_user: Annotated[User, Depends(get_current_active_user)]):
    try:
        # Gemini 的工具调用通常在聊天会话中进行管理
        chat = model.start_chat()
        
        # Step 1: 将用户指令发送给 LLM
        response = chat.send_message(request.user_command)
        
        response_part = response.candidates[0].content.parts[0]
        
        # Step 2: 检查 LLM 是否决定调用工具
        if response_part.function_call:
            function_call = response_part.function_call
            function_name = function_call.name
            function_args = {key: value for key, value in function_call.args.items()}
            
            print(f"LLM 决定调用工具：{function_name}，参数：{function_args}")
            
            # 根据函数名调用本地的工具函数
            if function_name == "get_current_weather":
                tool_output = get_current_weather(**function_args)
            else:
                raise HTTPException(status_code=400, detail=f"未知的工具函数: {function_name}")

            # Step 3: 将工具的输出结果再次发送给 LLM，让它生成最终的自然语言响应
            final_response = chat.send_message(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=function_name,
                        # Gemini 需要一个包含 "output" 键的字典作为响应
                        response={"output": json.dumps(tool_output, ensure_ascii=False)}
                    )
                ),
            )
            
            return {"status": "success", "response": final_response.candidates[0].content.parts[0].text}
        else:
            # LLM 认为不需要调用工具，直接返回其响应
            return {"status": "success", "response": response.text}

    except Exception as e:
        print(f"发生错误: {e}")
        raise HTTPException(status_code=500, detail=str(e))