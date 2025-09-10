# Jenkins CI/CD 部署配置指南

## 1. Jenkins 环境准备

### 必需插件
- Docker Pipeline Plugin
- Docker Plugin
- SSH Agent Plugin
- Credentials Binding Plugin

### 系统配置
确保 Jenkins 服务器已安装：
- Docker
- Docker Compose
- Git

## 2. 凭证配置

### 2.1 Docker 私有仓库凭证
1. 进入 Jenkins → Manage Jenkins → Manage Credentials
2. 添加新凭证：
   - Kind: Username with password
   - ID: `docker-registry-credentials`
   - Username: 您的 Docker 仓库用户名
   - Password: 您的 Docker 仓库密码

### 2.2 WSL 服务器 SSH 凭证
1. 添加新凭证：
   - Kind: Username with password
   - ID: `wsl-server-credentials`
   - Username: WSL 服务器用户名
   - Password: WSL 服务器密码

## 3. 创建 Pipeline 任务

### 3.1 创建新任务
1. 点击 "New Item"
2. 选择 "Pipeline"
3. 输入任务名称：`susheng-ai-demo-pro`

### 3.2 配置 Pipeline
1. 在 Pipeline 配置中：
   - Definition: Pipeline script from SCM
   - SCM: Git
   - Repository URL: 您的 Git 仓库地址
   - Branch: `*/main` 或 `*/master`
   - Script Path: `CICD/pro/Jenkinsfile`

### 3.3 构建触发器
- 选择 "GitHub hook trigger for GITScm polling"
- 或选择 "Poll SCM" 设置定时构建

## 4. 服务器端配置

### 4.1 创建部署目录
在 WSL 服务器上执行：
```bash
mkdir -p /home/${USER}/workspace/docker
cd /home/${USER}/workspace/docker
```

### 4.2 复制部署文件
将以下文件复制到服务器：
- `CICD/deploy/docker-compose.yml`
- 创建 `.env` 文件并配置环境变量

### 4.3 设置权限
```bash
chmod +x /home/${USER}/workspace/docker/docker-compose.yml
```

## 5. 环境变量配置

### 5.1 必需的环境变量
在服务器的 `.env` 文件中配置：

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

## 6. 测试部署

### 6.1 手动触发构建
1. 进入 Jenkins 任务页面
2. 点击 "Build Now"
3. 查看构建日志

### 6.2 验证部署
部署成功后，访问以下地址：
- 应用首页: http://172.27.224.1:8080
- API 文档: http://172.27.224.1:8080/docs
- 健康检查: http://172.27.224.1:8080/index

## 7. 监控和维护

### 7.1 查看容器状态
```bash
docker-compose ps
```

### 7.2 查看应用日志
```bash
docker-compose logs -f susheng-ai-app
```

### 7.3 重启服务
```bash
docker-compose restart susheng-ai-app
```

## 8. 故障排除

### 8.1 常见问题
1. **Docker 镜像拉取失败**
   - 检查 Docker 仓库连接
   - 验证凭证配置

2. **容器启动失败**
   - 检查环境变量配置
   - 查看容器日志

3. **数据库连接失败**
   - 确认 MySQL 服务状态
   - 检查网络连接

4. **API 调用失败**
   - 验证 OpenAI API Key
   - 检查网络访问权限

### 8.2 日志查看
- Jenkins 构建日志：在任务页面查看
- 容器运行日志：`docker-compose logs`
- 应用日志：`./logs/fastapi.log`
