// 简化的 Jenkins Pipeline 脚本
pipeline {
    agent any

    environment {
        DOCKER_IMAGE = 'susheng-ai-demo:pro'
        DOCKER_CONTAINER = 'susheng-ai-app'
        MYSQL_HOST = '172.27.224.1'
        MYSQL_PORT = '3306'
    }

    stages {
        stage('Build & Deploy') {
            steps {
                script {
                    // 停止并删除旧容器
                    sh 'docker stop $DOCKER_CONTAINER 2>/dev/null || true'
                    sh 'docker rm $DOCKER_CONTAINER 2>/dev/null || true'
                    
                    // 构建新镜像
                    sh 'docker build -t $DOCKER_IMAGE .'
                    
                    // 启动新容器
                    sh 'docker run -d --name $DOCKER_CONTAINER -p 8080:8000 -e MYSQL_HOST=$MYSQL_HOST -e MYSQL_PORT=$MYSQL_PORT $DOCKER_IMAGE'
                }
            }
        }
    }

    post {
        success {
            echo '✅ 部署成功!'
            echo '🌐 应用地址: http://localhost:8080'
            echo '📚 API文档: http://localhost:8080/docs'
        }
        failure {
            echo '❌ 部署失败，请检查日志'
        }
    }
}
