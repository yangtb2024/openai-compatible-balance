FROM python:3.9-slim

WORKDIR /app

# 复制所需文件到容器中
COPY ./app /app/app
COPY ./main.py /app
COPY ./requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt
ENV API_KEYS=["your_api_key_1"]
ENV ALLOWED_TOKENS=["your_token_1"]
ENV BASE_URL=https://api.groq.com/openai/v1

# Expose port
EXPOSE 7860

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
