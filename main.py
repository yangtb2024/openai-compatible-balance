from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
from typing import List, Optional
import logging
from itertools import cycle
import asyncio

import uvicorn

from app import config

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API密钥配置
API_KEYS = config.settings.API_KEYS

# 创建一个循环迭代器
key_cycle = cycle(API_KEYS)
key_lock = asyncio.Lock()


class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "llama-3.2-90b-text-preview"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 8000
    stream: Optional[bool] = False


async def verify_authorization(authorization: str = Header(None)):
    if not authorization:
        logger.error("Missing Authorization header")
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        logger.error("Invalid Authorization header format")
        raise HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )
    token = authorization.replace("Bearer ", "")
    if token not in config.settings.ALLOWED_TOKENS:
        logger.error("Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")
    return token


@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    await verify_authorization(authorization)
    async with key_lock:
        api_key = next(key_cycle)
        logger.info(f"Using API key: {api_key[:8]}...")
    try:
        client = openai.OpenAI(api_key=api_key, base_url=config.settings.BASE_URL)
        response = client.models.list()
        logger.info("Successfully retrieved models list")
        return response
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest, authorization: str = Header(None)):
    await verify_authorization(authorization)
    async with key_lock:
        api_key = next(key_cycle)
        logger.info(f"Using API key: {api_key[:8]}...")

    try:
        logger.info(f"Chat completion request - Model: {request.model}")
        client = openai.OpenAI(api_key=api_key, base_url=config.settings.BASE_URL)
        response = client.chat.completions.create(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream if hasattr(request, "stream") else False,
        )

        if hasattr(request, "stream") and request.stream:
            logger.info("Streaming response enabled")

            async def generate():
                for chunk in response:
                    yield f"data: {chunk.model_dump_json()}\n\n"

            return StreamingResponse(content=generate(), media_type="text/event-stream")

        logger.info("Chat completion successful")
        return response

    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(authorization: str = Header(None)):
    await verify_authorization(authorization)
    logger.info("Health check endpoint called")
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)