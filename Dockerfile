FROM python:3.9-slim

WORKDIR /app

# 复制所需文件到容器中
COPY ./app /app/app
COPY ./main.py /app
COPY ./requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
