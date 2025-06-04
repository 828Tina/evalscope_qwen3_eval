from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
from transformers.generation.utils import GenerationConfig
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi import FastAPI,Query
from pydantic import BaseModel, Field
import torch.nn.functional as F
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Tuple, Union
import time

######################
# 定义FastAPI应用程序
######################
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

######################
# model+tokenizer加载
######################

# 确保设备是可用的
device = "cuda:0"  
# 加载模型和分词器
model_path = "/data/nvme1/weights/Qwen3_sft_eval/output"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)
model.eval()  # 确保模型处于评估模式

######################
# 定义FastAPI请求响应格式
######################

# 定义OpenAI的content格式
class TextContent(BaseModel):
    type: Literal["text"]
    text: str

# 定义OpenAI的content格式
ContentItem = Union[TextContent]

# 定义OpenAI的message格式
class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None

# 定义OpenAI的api格式的输出message相映格式
class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

# 定义OpenAI的api格式的模型输入
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

# 定义OpenAI的api格式的输出chattemplate选择
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse
class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

# 定义OpenAI的api格式的模型输出
class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

######################
# 处理请求并生成
######################
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    该部分代码是将SFT模型部署到FastAPI上，并提供一个API接口，用于处理用户输入的文本，并返回模型生成的文本。

    1. 首先，定义了两个全局变量：model 和 tokenizer，分别用于加载和处理模型。
    2. 然后从request中提取信息，用于后续的模型推理生成，也就是将处理部分在服务器上
    """
    global model,tokenizer

    # 默认参数
    enable_thinking: bool = False
    # openai格式输入
    messages = request.messages
    temperature = request.temperature if request.temperature is not None else 0.8
    max_tokens = request.max_tokens if request.max_tokens is not None else 2048

    # 将输入消息转换为模型输入格式
    text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=enable_thinking # 是否启用思考模式
    )
    
    # 编码输入文本
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # 生成响应
    with torch.no_grad():
        generation_ids = model.generate(
                                        **model_inputs,
                                        temperature=temperature,
                                        max_new_tokens=max_tokens,
                                        )
    output_ids = generation_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    # 解码输出文本
    response_texts = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    # 构建响应对象
    choices = [
        ChatCompletionResponseChoice(
            index=0,
            message=ChatMessageResponse(role="assistant", content=response_texts)
        )
    ]
    # 计算使用信息
    usage = UsageInfo()
    # 如果是流式响应，则使用streaming的格式
    response = ChatCompletionResponse(
        model=request.model,
        object="chat.completion",
        choices=choices,
        usage=usage,
        )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=25001)
    """
    curl -X POST "http://127.0.0.1:25001/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{
            "model":"qwen3_sft_eval",
            "messages": [
                {
                "role": "user",
                "content": "三原色是什么？"
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.9
            }'

    """