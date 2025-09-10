# 生产环境部署说明

## 部署前准备

### 1. 服务器环境要求
- Docker 和 Docker Compose 已安装
- 端口 8080 可用
- 能够访问外部 MySQL 数据库

### 2. 环境变量配置
在服务器上创建 `.env` 文件，配置以下环境变量：

```bash
# MySQL 数据库配置
MYSQL_HOST=172.27.224.1
MYSQL_PORT=3306
MYSQL_USERNAME=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=chat_xz

# OpenAI API 配置
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_MODEL=gpt-4o

# Quick API 配置（可选）
QUICK_API_KEY=your_quick_api_key
QUICK_API_BASE=your_quick_api_base
QUICK_API_MODEL=gpt-4o
```

### 3. 目录结构
确保服务器上的目录结构如下：
```
/home/${USER}/workspace/docker/
├── docker-compose.yml
├── .env
└── logs/
```

## Jenkins 部署流程

### 1. Jenkins 配置
- 确保 Jenkins 已安装 Docker 插件
- 配置以下凭证：
  - `docker-registry-credentials`: Docker 私有仓库凭证
  - `wsl-server-credentials`: WSL 服务器 SSH 凭证

### 2. 部署步骤
1. 代码推送到 Git 仓库
2. Jenkins 自动触发构建
3. 构建 Docker 镜像
4. 推送到私有仓库
5. 在 WSL 服务器上部署

### 3. 访问地址
- 应用首页: http://172.27.224.1:8080
- API 文档: http://172.27.224.1:8080/docs
- 健康检查: http://172.27.224.1:8080/index

## 手动部署

如果需要手动部署，可以执行以下命令：

```bash
# 1. 拉取最新镜像
docker pull 172.27.224.1:5001/susheng-ai-demo:pro

# 2. 停止旧容器
docker-compose down

# 3. 启动新容器
docker-compose up -d

# 4. 查看日志
docker-compose logs -f
```

## 故障排除

### 1. 容器启动失败
- 检查环境变量配置
- 检查 MySQL 连接
- 查看容器日志：`docker-compose logs susheng-ai-app`

### 2. 数据库连接失败
- 确认 MySQL 服务运行正常
- 检查网络连接
- 验证数据库用户权限

### 3. API 调用失败
- 检查 OpenAI API Key 配置
- 确认网络可以访问外部 API
