# 🚀 FastAPI OpenAI 代理服务

## 📝 项目简介

这是一个基于 FastAPI 框架开发的 OpenAI API 代理服务，支持多 API Key 轮询和流式响应。

## ✨ 主要特性

- 🔄 多 API Key 轮询支持
- 🔐 Bearer Token 认证
- 📡 支持流式响应
- 🌐 CORS 跨域支持
- 📊 健康检查接口

## 🛠️ 技术栈

- FastAPI
- OpenAI
- Pydantic
- Docker

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Docker (可选)

### 📦 安装依赖

```bash
pip install -r requirements.txt
```

### ⚙️ 配置文件

创建 `.env` 文件并配置以下参数：

```env
API_KEYS=["your-api-key-1","your-api-key-2"]
ALLOWED_TOKENS=["your-access-token-1","your-access-token-2"]
BASE_URL="https://api.openai.com/v1"
```

### 🐳 Docker 部署

```bash
docker build -t fastapi-openai-proxy .
docker run -p 8000:8000 -d fastapi-openai-proxy
```

## 🔌 API 接口

### 获取模型列表

```http
GET /v1/models
Authorization: Bearer your-token
```

### 聊天完成

```http
POST /v1/chat/completions
Authorization: Bearer your-token

{
    "messages": [...],
    "model": "llama-3.2-90b-text-preview",
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": false
}
```

### 健康检查

```http
GET /health
Authorization: Bearer your-token
```

## 📚 代码结构

- `app/main.py`: 主应用程序入口
- `app/config.py`: 配置管理
- `Dockerfile`: 容器化配置
- `requirements.txt`: 项目依赖

## 🔒 安全特性

- API Key 轮询机制
- Bearer Token 认证
- 请求日志记录

## 📝 注意事项

- 请确保妥善保管 API Keys 和访问令牌
- 建议在生产环境中使用环境变量配置敏感信息
- 默认服务端口为 8000

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
