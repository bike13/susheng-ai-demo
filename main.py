# -*- coding: utf-8 -*-
import os
import sys
import tempfile
import whisper
import warnings
import time

# 设置默认编码为UTF-8，避免Windows系统GBK编码问题
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
from fastapi import FastAPI, Body, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any, List, Optional, Generator
# 移除不再需要的langgraph导入，因为现在使用MySQL而不是Redis检查点
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from prompt import prompt_dict
import logging
import dotenv
import json
import redis
import uuid
from datetime import datetime
import openai
import pymysql

# 过滤 FP16 相关警告
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# 创建FastAPI实例
app = FastAPI(
    title="聊天API服务",
    description="基于OpenAI和Redis的智能聊天API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/logs")
async def get_log_file():
    """
    输出日志文件内容

    Returns:
        FileResponse: 日志文件的下载响应
    """
    log_file_path = os.getenv("LOG_FILE_PATH", "app.log")
    if not os.path.exists(log_file_path):
        return JSONResponse(content={"success": False, "error": "日志文件不存在"}, status_code=404)
    return FileResponse(log_file_path, media_type="text/plain", filename=os.path.basename(log_file_path))

# 加载环境变量配置
dotenv.load_dotenv()

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# 确保日志处理器使用UTF-8编码
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
logger = logging.getLogger(__name__)



class ShortTermMemoryMySQL:
    """基于MySQL的短期记忆实现"""

    def __init__(self, mysql_config: Optional[Dict[str, Any]] = None, system_prompt: Optional[str] = None):
        """
        初始化ShortTermMemoryMySQL实例
        
        Args:
            mysql_config: MySQL连接配置，如果未提供则从环境变量获取
            system_prompt: 系统提示词，如果未提供则从环境变量获取
        """
        # MySQL配置
        self.mysql_config = mysql_config or {
            'host': os.getenv("MYSQL_HOST", "localhost"),
            'port': int(os.getenv("MYSQL_PORT", "3306")),
            'user': os.getenv("MYSQL_USERNAME", "root"),
            'password': os.getenv("MYSQL_PASSWORD", ""),
            'database': os.getenv("MYSQL_DATABASE", "chat_xz"),
            'charset': 'utf8mb4'
        }
        
        # OpenAI配置
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_base = os.getenv("OPENAI_API_BASE")
        self.openai_api_model = os.getenv("OPENAI_API_MODEL", "gpt-4o")
        self.system_prompt = system_prompt or prompt_dict["system_prompt"]
        
        self.connection = None
        self.model = None

    def setup(self):
        """
        设置MySQL连接和模型
        
        初始化MySQL连接、创建数据库表、初始化OpenAI模型。
        必须在调用其他方法之前调用此方法。
        
        Raises:
            ValueError: 当OPENAI_API_KEY环境变量未设置时
        """
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY 环境变量未设置")

        # 初始化MySQL连接
        self._init_mysql_connection()
        
        # 创建数据库表
        self._create_tables()

        # 初始化OpenAI模型
        self.model = ChatOpenAI(
            model=self.openai_api_model,
            api_key=self.openai_api_key,
            base_url=self.openai_api_base
        )

        logger.info("MySQL短期记忆设置完成")

    def _init_mysql_connection(self):
        """初始化MySQL连接"""
        try:
            self.connection = pymysql.connect(**self.mysql_config)
            logger.info(f"MySQL连接初始化成功: {self.mysql_config['host']}:{self.mysql_config['port']}")
        except Exception as e:
            logger.error(f"MySQL连接失败: {e}")
            raise

    def _create_tables(self):
        """创建数据库表"""
        try:
            with self.connection.cursor() as cursor:
                # 创建会话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id int AUTO_INCREMENT COMMENT '主键id',
                        session_id varchar(64) COMMENT '会话id',
                        user_id varchar(64) DEFAULT NULL COMMENT '用户id',
                        session_name varchar(64) DEFAULT NULL COMMENT '会话名称',
                        create_time datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        update_time datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                        PRIMARY KEY (`id`),
                        KEY `idx_user_id` (`user_id`),
                        KEY `idx_session_id` (`session_id`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='会话表'
                """)
                
                # 创建对话表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id int AUTO_INCREMENT COMMENT '主键id',
                        session_id varchar(64) COMMENT '会话id',
                        user_id varchar(64) NOT NULL COMMENT '用户id',
                        role varchar(64) COMMENT '角色',
                        content text COMMENT '内容',
                        create_time datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        update_time datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                        PRIMARY KEY (`id`),
                        KEY `idx_session_id` (`session_id`),
                        KEY `idx_user_id` (`user_id`),
                        KEY `idx_session_user` (`session_id`, `user_id`)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='对话表'
                """)
                
                self.connection.commit()
                logger.info("数据库表创建成功")
        except Exception as e:
            logger.error(f"创建数据库表失败: {e}")
            raise

    def _ensure_session_exists(self, session_id: str, user_id: Optional[str] = None) -> bool:
        """确保会话存在，如果不存在则创建"""
        try:
            with self.connection.cursor() as cursor:
                # 检查会话是否存在
                cursor.execute("SELECT id FROM chat_sessions WHERE session_id = %s", (session_id,))
                if cursor.fetchone():
                    return True
                
                # 创建新会话
                cursor.execute("""
                    INSERT INTO chat_sessions (session_id, user_id, session_name, create_time, update_time)
                    VALUES (%s, %s, %s, NOW(), NOW())
                """, (session_id, user_id, f"会话_{session_id[:8]}"))
                
                self.connection.commit()
                logger.info(f"创建新会话: {session_id}")
                return True
        except Exception as e:
            logger.error(f"会话存在失败: {e}")
            return False

    def _save_message(self, session_id: str, user_id: str, role: str, content: str) -> bool:
        """保存消息到数据库"""
        try:
            logger.info(f"保存消息到数据库: session_id={session_id}, user_id={user_id}, role={role}")
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO chat_messages (session_id, user_id, role, content, create_time, update_time)
                    VALUES (%s, %s, %s, %s, NOW(), NOW())
                """, (session_id, user_id, role, content))
                
                self.connection.commit()
                return True
        except Exception as e:
            logger.error(f"保存消息失败: {e}")
            return False

 
    def chat(self, message: str, thread_id: str, user_id: str, system_prompt: Optional[str] = None, historical_messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        进行对话
        
        Args:
            message: 用户输入的消息
            thread_id: 会话线程ID
            user_id: 用户ID
            system_prompt: 可选的系统提示词，覆盖默认提示词
            historical_messages: 可选的历史会话记录列表
            
        Returns:
            AI的响应内容
        """
        if not self.connection or not self.model:
            raise RuntimeError("请先调用setup()方法初始化")

        try:
            # 确保会话存在
            self._ensure_session_exists(thread_id, user_id)
            
            # 保存用户消息
            self._save_message(thread_id, user_id, "user", message)
            
            # 构建消息列表
            messages = []
            current_system_prompt = system_prompt or self.system_prompt
            if current_system_prompt:
                messages.append(SystemMessage(content=current_system_prompt))
            
            # 添加历史会话记录
            if historical_messages:
                for hist_msg in historical_messages:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
                logger.info(f"添加了 {len(historical_messages)} 条历史消息到上下文")
            else:
                # 从数据库获取历史消息
                db_history = self.get_conversation_history(thread_id, user_id)
                for hist_msg in db_history:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
                logger.info(f"从数据库加载了 {len(db_history)} 条历史消息")
            
            # 添加当前用户消息
            messages.append(HumanMessage(content=message))
            
            logger.info(f"构建了包含 {len(messages)} 条消息的完整上下文")
            
            # 调用AI模型
            response = self.model.invoke(messages)
            
            if response and hasattr(response, 'content'):
                ai_response = response.content
                # 保存AI响应
                self._save_message(thread_id, user_id, "assistant", ai_response)
                return ai_response

            return "抱歉，无法生成响应"

        except Exception as e:
            logger.error(f"对话失败: {e}")
            return f"对话过程中出现错误: {str(e)}"


    # 获取对话历史
    def get_conversation_history(self, thread_id: str, user_id: str) -> List[Dict[str, str]]:
        """获取对话历史"""
        if not self.connection:
            raise RuntimeError("数据库连接未初始化")

        try:
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT role, content, create_time 
                    FROM chat_messages 
                    WHERE session_id = %s AND user_id = %s
                    ORDER BY create_time ASC
                """, (thread_id, user_id))
                
                results = cursor.fetchall()
                history = []
                for row in results:
                    history.append({
                        "role": row[0],
                        "content": row[1],
                        "create_time": row[2].isoformat() if row[2] else None
                    })
                
                logger.info(f"从数据库获取了 {len(history)} 条历史消息 (用户: {user_id})")
                return history

        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []


    def suggested_chat(self, message: str, thread_id: str, user_id: str, system_prompt: Optional[str] = None, historical_messages: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        直接调用大模型生成推荐问题，不使用工作流
        强制输出JSON格式
        """
        if not self.connection:
            self.setup()

        try:
            # 构建历史对话记录字符串
            history_text = "用户对话记录：\n"
            if historical_messages:
                for hist_msg in historical_messages:
                    role = hist_msg.get("role", "")
                    content = hist_msg.get("content", "")
                    if role == "user":
                        history_text += f"用户：{content}\n"
                    elif role == "assistant":
                        history_text += f"助手：{content}\n"
            else:
                history_text += "暂无对话记录\n"
            
            # 构建消息列表
            prompt = prompt_dict.get("suggested_prompt", "")
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"{history_text}\n{message}"}
            ]


            
            # 设置OpenAI客户端
            client = openai.OpenAI(
                api_key=os.getenv("QUICK_API_KEY"),
                base_url=os.getenv("QUICK_API_BASE")
            )
            
            # 调用OpenAI API，强制输出JSON格式
            response = client.chat.completions.create(
                model=os.getenv("QUICK_API_MODEL", "gpt-4o"),
                messages=messages,
                response_format={"type": "json_object"},
                stream=False,
            )
            
            # 提取AI响应内容
            return response.choices[0].message.content



        except Exception as e:
            logger.error(f"推荐问题生成失败: {e}")
            # 返回默认的JSON格式
            return '{"questions": ["请介绍一下你自己", "你能做什么？", "如何使用这个AI助手？"]}'

    def chat_stream(self, message: str, thread_id: str, system_prompt: Optional[str] = None, historical_messages: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None) -> Generator[str, None, None]:
        """
        流式对话
        
        Args:
            message: 用户输入的消息
            thread_id: 会话线程ID
            system_prompt: 可选的系统提示词，覆盖默认提示词
            historical_messages: 可选的历史会话记录列表
            user_id: 用户ID
            
        Yields:
            AI响应的内容片段
        """
        if not self.connection or not self.model:
            raise RuntimeError("请先调用setup()方法初始化")

        try:
            # 确保会话存在
            self._ensure_session_exists(thread_id, user_id)
            
            # 保存用户消息
            self._save_message(thread_id, user_id, "user", message)
            
            # 构建消息列表
            messages = []
            current_system_prompt = system_prompt or self.system_prompt
            if current_system_prompt:
                messages.append(SystemMessage(content=current_system_prompt))
            
            # 添加历史会话记录
            if historical_messages:
                for hist_msg in historical_messages:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
                logger.info(f"添加了 {len(historical_messages)} 条历史消息到流式对话上下文")
            else:
                # 从数据库获取历史消息
                db_history = self.get_conversation_history(thread_id, user_id)
                for hist_msg in db_history:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
                logger.info(f"从数据库加载了 {len(db_history)} 条历史消息到流式对话")
            
            # 添加当前用户消息
            messages.append(HumanMessage(content=message))
            
            logger.info(f"构建了包含 {len(messages)} 条消息的完整流式对话上下文")
            
            # 流式调用AI模型
            full_response = ""
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            # 保存完整的AI响应
            if full_response:
                self._save_message(thread_id, user_id, "assistant", full_response)

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield f"流式对话过程中出现错误: {str(e)}"
    
    def chat_stream_sse(self, message: str, thread_id: str, system_prompt: Optional[str] = None, historical_messages: Optional[List[Dict[str, Any]]] = None, user_id: Optional[str] = None) -> Generator[str, None, None]:
        """
        进行SSE格式的流式对话
        
        Args:
            message: 用户输入的消息
            thread_id: 会话线程ID
            system_prompt: 可选的系统提示词，覆盖默认提示词
            historical_messages: 可选的历史会话记录列表
            user_id: 用户ID
            
        Yields:
            SSE格式的响应字符串
        """
        if not self.connection or not self.model:
            raise RuntimeError("请先调用setup()方法初始化")

        try:
            # 确保会话存在
            self._ensure_session_exists(thread_id, user_id)
            
            # 保存用户消息
            logger.info(f"准备保存用户消息: session_id={thread_id}, user_id={user_id}, message={message[:50]}...")
            self._save_message(thread_id, user_id, "user", message)
            
            # 构建消息列表
            messages = []
            current_system_prompt = system_prompt or self.system_prompt
            if current_system_prompt:
                messages.append(SystemMessage(content=current_system_prompt))
            
            # 添加历史会话记录
            if historical_messages:
                for hist_msg in historical_messages:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
            else:
                # 从数据库获取历史消息
                db_history = self.get_conversation_history(thread_id, user_id)
                for hist_msg in db_history:
                    if hist_msg.get("role") == "user":
                        messages.append(HumanMessage(content=hist_msg.get("content", "")))
                    elif hist_msg.get("role") == "assistant":
                        messages.append(AIMessage(content=hist_msg.get("content", "")))
            
            # 添加当前用户消息
            messages.append(HumanMessage(content=message))
            
            # 直接使用模型进行流式生成
            full_response = ""
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    # 返回SSE格式的数据
                    yield f"data: {json.dumps({'chunk': chunk.content})}\n\n"
            
            # 生成结束后，推送一个特殊的"[DONE]"标记
            yield f"data: {json.dumps({'chunk': '[DONE]'})}\n\n"
            
            # 重要：保存对话历史到MySQL
            try:
                if full_response:
                    self._save_message(thread_id, user_id, "assistant", full_response)
                logger.info(f"对话历史已保存到MySQL，会话ID: {thread_id}")
            except Exception as e:
                logger.warning(f"保存对话历史失败: {e}")

        except Exception as e:
            logger.error(f"SSE流式对话失败: {e}")
            error_msg = f"流式响应错误: {str(e)}"
            yield f"data: {json.dumps({'chunk': error_msg})}\n\n"

    def delete_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        删除指定session_id的MySQL会话数据
        
        Args:
            session_id: 要删除的会话ID
            user_id: 用户ID
            
        Returns:
            包含删除结果的字典
        """
        try:
            if not self.connection:
                self.setup()
            
            with self.connection.cursor() as cursor:
                # 删除该会话的所有消息（按用户ID过滤）
                cursor.execute("DELETE FROM chat_messages WHERE session_id = %s AND user_id = %s", (session_id, user_id))
                deleted_messages = cursor.rowcount
                
                # 删除会话记录（按用户ID过滤）
                cursor.execute("DELETE FROM chat_sessions WHERE session_id = %s AND user_id = %s", (session_id, user_id))
                deleted_sessions = cursor.rowcount
                
                self.connection.commit()
                
                logger.info(f"删除会话 {session_id} (用户: {user_id}) 完成: 删除了 {deleted_messages} 条消息和 {deleted_sessions} 个会话记录")
                
                return {
                    "success": True,
                    "message": f"Session {session_id} 及其历史会话记录已删除，共删除 {deleted_messages} 条消息和 {deleted_sessions} 个会话记录"
                }
            
        except Exception as e:
            logger.error(f"删除会话数据失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"删除Session {session_id} 失败"
            }

    def get_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """
        获取指定session_id的MySQL会话数据
        
        Args:
            session_id: 要查询的会话ID
            
        Returns:
            包含会话数据的响应字典
        """
        try:
            if not self.connection:
                self.setup()
            
            logger.info(f"获取MySQL中的 session_id: {session_id}")

            # 从数据库获取历史消息
            history = self.get_conversation_history(session_id, user_id)
            
            result_messages = []
            if history:
                # 最多返回最新的10轮对话
                max_rounds = 10
                total_rounds = len(history) // 2
                
                if total_rounds > max_rounds:
                    # 从最新的消息开始，取最新的10轮对话
                    start_index = len(history) - (max_rounds * 2)
                    messages_to_process = history[start_index:]
                    logger.info(f"限制为最新 {max_rounds} 轮对话，从第 {start_index + 1} 条消息开始")
                else:
                    messages_to_process = history
                    logger.info(f"对话轮数不超过 {max_rounds}，返回全部对话")
                
                for idx, msg in enumerate(messages_to_process):
                    msg_id = f"msg_{str(idx+1).zfill(3)}"
                    result_messages.append({
                        "id": msg_id,
                        "role": msg.get("role"),
                        "content": msg.get("content"),
                        "timestamp": msg.get("create_time")
                    })

                logger.info(f"成功获取到 {len(result_messages)} 条对话历史（{len(result_messages)//2} 轮对话）")

                for msg in result_messages:
                    role = msg.get("role")
                    content = msg.get("content", "")
                    if role == "assistant" and isinstance(content, str) and len(content) > 30:
                        # 安全处理Unicode字符，避免编码错误
                        safe_content = content[:30].encode('utf-8', errors='ignore').decode('utf-8')
                        logger.info(f"历史会话[{role}]: {safe_content}...（共{len(content)}字）")
                    else:
                        # 安全处理Unicode字符，避免编码错误
                        safe_content = content.encode('utf-8', errors='ignore').decode('utf-8')
                        logger.info(f"历史会话[{role}]: {safe_content}")
            else:
                logger.info("未获取到消息内容")

            response = {
                "success": True,
                "data": {
                    "session_id": session_id,
                    "messages": result_messages,
                    "has_more": False
                }
            }
            return response
            
        except Exception as e:
            logger.error(f"获取会话数据失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "data": {
                    "session_id": session_id,
                    "messages": [],
                    "has_more": False
                }
            }


    # 创建会话
    def create_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """创建新的会话"""
        try:
            if not self.connection:
                self.setup()
            with self.connection.cursor() as cursor:
                # 向 chat_sessions 表中插入一条新会话记录，包含会话ID、用户ID和会话名称（会话名称为"会话_"加上会话ID的前8位）
                cursor.execute(
                    "INSERT INTO chat_sessions (session_id, user_id, session_name) VALUES (%s, %s, %s)",
                    (session_id, user_id, "新会话_" + datetime.now().strftime("%Y%m%d%H%M%S"))
                )
                self.connection.commit()
                return {
                    "success": True,
                    "message": "会话创建成功",
                    "data": {
                        "session_id": session_id
                    }
                }
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "创建会话失败"
            }


    def close(self):
        """关闭MySQL连接"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("MySQL连接已关闭")
            except Exception as e:
                logger.warning(f"关闭MySQL连接时出现警告: {e}")
            finally:
                self.connection = None
        
    def __del__(self):
        """析构函数，确保资源正确释放"""
        try:
            self.close()
        except Exception:
            pass  # 忽略析构函数中的异常


@app.get("/")
async def show_index():
    """
    首页接口
    
    返回聊天界面HTML页面。
    
    Returns:
        HTML页面文件
    """
    return FileResponse("static/chat.html")

@app.get("/index")
async def show_index_api():
    """
    API首页接口
    
    返回API服务的基本信息，用于健康检查。
    
    Returns:
        包含欢迎消息的字典
    """
    return {"message": "这是首页!"}


# ==================== Chat API 路由 ====================
# 以下路由实现了完整的聊天功能，包括会话管理、消息发送和历史记录查询

@app.post("/chat/sessions")
async def create_chat_session(request: dict = Body(...)):
    # 验证必需参数
    user_id = request.get("user_id")
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    try:
        session_id = str(uuid.uuid4()).replace('-', '')
        # 使用ShortTermMemoryMySQL类的create_session方法
        memory = ShortTermMemoryMySQL()
        memory.setup()  # 初始化数据库连接
        result = memory.create_session(session_id, user_id)
        return result
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "创建会话失败"
        }
        


@app.post("/knowledge/analysis")
async def get_knowledge_analysis(request: dict = Body(...)):
    subject = request.get("subject")
    knowledge = request.get("knowledge")
    if not subject:
        return {
            "success": False,
            "error": "subject is required",
            "message": "subject参数是必需的"
        }
    if not knowledge:
        return {
            "success": False,
            "error": "knowledge is required",
            "message": "knowledge参数是必需的"
        }
    try:
        system_prompt= prompt_dict["knowledge_prompt"]
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )

        messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"subject:{subject}\nknowledge_name:{knowledge}"}
            ]

        # 调用大模型
        chat_completion = client.chat.completions.create(
            model=os.getenv("QUICK_API_MODEL"),
            messages=messages,
            response_format={"type": "json_object"},
            stream=False
        )

        # 解析大模型返回的JSON字符串，调整返回格式
        response_content = chat_completion.choices[0].message.content
        try:
            # 尝试解析为JSON对象
            data = json.loads(response_content)
            # 可选：打印解析后的内容，便于调试
            logger.info(f"知识点分析内容: {data}")
        except Exception as parse_error:
            return {
                "success": False,
                "error": "知识点分析内容解析失败",
                "message": str(parse_error),
                "raw_content": response_content
            }

        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        # 安全处理错误信息，避免编码问题
        error_msg = str(e).encode('utf-8', errors='ignore').decode('utf-8')
        logger.error(f"获取知识点分析失败: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "message": "获取知识点分析失败"
        }


@app.post("/chat/history")
async def get_chat_history(request: dict = Body(...)):
    # 验证必需参数
    session_id = request.get("session_id")
    user_id = request.get("user_id")
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required",
            "message": "session_id参数是必需的"
        }
    
    try:
        # 使用ShortTermMemoryMySQL类的get_session方法
        memory = ShortTermMemoryMySQL()
        result = memory.get_session(session_id, user_id)
        return result
    except Exception as e:
        logger.error(f"获取会话历史失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "session_id": session_id,
                "messages": [],
                "has_more": False
            }
        }


@app.post("/chat/messages")
async def send_chat_message(request: dict = Body(...)):
    # 提取和验证请求参数
    session_id = request.get("session_id")
    user_id = request.get("user_id")
    message = request.get("message")
    
    # 验证必需参数
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required",
            "message": "session_id参数是必需的"
        }
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    if not message:
        return {
            "success": False,
            "error": "message is required",
            "message": "message参数是必需的"
        }
    # 验证消息不为空
    if not message.strip():
        return {
            "success": False,
            "error": "message cannot be empty",
            "message": "消息内容不能为空"
        }
    
    try:
        system_prompt= prompt_dict["system_prompt"]
        # 1. 初始化AI记忆系统
        memory = ShortTermMemoryMySQL(system_prompt=system_prompt)
        memory.setup()
        
        # 2. 获取历史会话记录
        logger.info(f"获取会话 {session_id} 的历史记录")
        history_response = memory.get_session(session_id, user_id)
        historical_messages = []
        
        if history_response.get("success") and history_response.get("data", {}).get("messages"):
            historical_messages = history_response["data"]["messages"]
            logger.info(f"成功获取到 {len(historical_messages)} 条历史消息")
        else:
            logger.info("未获取到历史消息或会话不存在")
        
        # 3. 调用AI模型生成响应，传入历史会话记录
        response = memory.chat(message, session_id, user_id, system_prompt, historical_messages)
        
        # 4. 返回成功响应
        return {
            "success": True,
            "data": {
                "session_id": session_id,
                "response": response,
                "timestamp": datetime.now().isoformat() + "Z"
            }
        }
    except Exception as e:
        logger.error(f"发送消息失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "发送消息失败"
        }


@app.post("/chat/messages/stream")
async def send_chat_message_stream(request: dict = Body(...)):

    # 提取和验证请求参数
    session_id = request.get("session_id")
    user_id = request.get("user_id")
    message = request.get("message")
    
    # 验证必需参数
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required",
            "message": "session_id参数是必需的"
        }
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    if not message:
        return {
            "success": False,
            "error": "message is required",
            "message": "message参数是必需的"
        }

    try:
        system_prompt= prompt_dict["system_prompt"]
        # 1. 初始化AI记忆系统（ShortTermMemoryMySQL），用于管理会话状态和与AI模型的交互。
        memory = ShortTermMemoryMySQL(system_prompt=system_prompt)
        memory.setup()
        
        # 2. 获取历史会话记录
        logger.info(f"获取会话 {session_id} 的历史记录")
        history_response = memory.get_session(session_id, user_id)
        historical_messages = []
        
        if history_response.get("success") and history_response.get("data", {}).get("messages"):
            historical_messages = history_response["data"]["messages"]
            logger.info(f"成功获取到 {len(historical_messages)} 条历史消息")
        else:
            logger.info("未获取到历史消息或会话不存在")
        
        # 3. 使用ShortTermMemoryMySQL类的chat_stream_sse方法进行流式对话
        logger.info(f"开始流式对话，会话ID: {session_id}")
        
        # 4. 返回StreamingResponse对象，直接使用memory.chat_stream_sse方法
        return StreamingResponse(

            memory.chat_stream_sse(message, session_id, system_prompt, historical_messages, user_id),

            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*"
            }
        )
    except Exception as e:
        # 5. 如果初始化或整体流程出错，返回一个流式的错误响应。
        logger.error(f"流式发送消息失败: {e}")
        def error_stream():
            """错误流式响应"""
            yield f"data: {json.dumps({'chunk': f'流式响应初始化失败: {str(e)}'})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )


@app.delete("/chat/deleteSessions")
async def delete_chat_session(request: dict = Body(...)):
    # 验证必需参数
    session_id = request.get("session_id")
    user_id = request.get("user_id")
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required",
            "message": "session_id参数是必需的"
        }
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    
    try:
        # 使用ShortTermMemoryMySQL类的delete_session方法
        memory = ShortTermMemoryMySQL()
        result = memory.delete_session(session_id, user_id)
        return result
    except Exception as e:
        logger.error(f"删除会话失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"删除Session {session_id} 失败"
        }


# 将这部分代码替换：
@app.get("/chat/suggestions")
async def get_chat_suggestions(session_id: str = Query(...), user_id: str = Query(...)):
    if not session_id:
        return {
            "success": False,
            "error": "session_id is required"
        }
    if not user_id:
        return {
            "success": False,
            "error": "user_id is required",
            "message": "user_id参数是必需的"
        }
    try:
        # 1. 初始化AI记忆系统
        memory = ShortTermMemoryMySQL(system_prompt=prompt_dict["suggested_prompt"])
        memory.setup()
        # 2. 获取历史会话记录
        history_response = memory.get_session(session_id, user_id)
        historical_messages = []
        if history_response.get("success") and history_response.get("data", {}).get("messages"):
            historical_messages = history_response["data"]["messages"]
            logger.info(f"成功获取到 {len(historical_messages)} 条历史消息")
        else:
            logger.info("未获取到历史消息或会话不存在")

        # 3. 调用AI模型生成推荐问题
        suggestions = memory.suggested_chat('请根据历史会话记录生成3个推荐问题', session_id, user_id, system_prompt=prompt_dict["suggested_prompt"], historical_messages=historical_messages)
        logger.info(f"推荐问题: {suggestions}")
        
        # 解析AI返回的JSON字符串
        try:
            if isinstance(suggestions, str):
                suggestions = json.loads(suggestions)
            
            # 检查questions字段是否存在且不为空
            if not suggestions or not suggestions.get("questions") or len(suggestions.get("questions", [])) == 0:
                logger.info("AI返回的推荐问题为空，使用默认问题")
                suggestions = {
                    "questions": [
                        "请介绍一下你自己",
                        "你能做什么？", 
                        "如何使用这个AI助手？"
                    ]
                }
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"解析AI返回的JSON失败: {e}，使用默认问题")
            suggestions = {
                "questions": [
                    "请介绍一下你自己",
                    "你能做什么？", 
                    "如何使用这个AI助手？"
                ]
            }

        return {
            "success": True,
            "data": {
                "suggestions": suggestions["questions"]
            }
        }
    except Exception as e:
        logger.error(f"获取推荐问题失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "data": {
                "suggestions": []
            }
        }


@app.post("/chat/voice")
async def get_chat_voice(file: UploadFile = File(...)):
    try:
        # 保存上传的音频文件到临时目录
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            audio_path = temp_audio.name
            logger.info(f"音频文件已保存到: {audio_path}")

        # 使用Whisper进行语音识别
        model = whisper.load_model("base") # 可以选择不同的模型，如 "tiny", "small", "medium", "large"
        result = model.transcribe(audio_path)
        logger.info(f"语音识别结果: {result['text']}")

        # 清理临时文件
        os.unlink(audio_path)
        logger.info(f"临时音频文件已删除: {audio_path}")

        return JSONResponse(content={"text": result["text"]})
    except Exception as e:
        logger.error(f"语音识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音识别失败: {str(e)}")


if __name__ == "__main__":
    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)