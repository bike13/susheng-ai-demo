# 使用官方的 Python 镜像
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
# 设置容器内的工作目录
WORKDIR /app

# 更新 pip
RUN pip install --upgrade pip

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 

# 复制整个 app 目录（包含子目录和静态文件）
COPY . .


# 暴露应用的端口
EXPOSE 8000
# 直接启动 FastAPI 应用
CMD ["python", "main.py"]
