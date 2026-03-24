import os
import httpx
from openai import AsyncOpenAI
from config import LLM_CONFIG

class LLMClient:
    """
    大模型客户端类，用于调用qwen-plus模型生成回复
    """
    
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        """
        初始化大模型客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        print("API Key 是否存在:", bool(api_key))
        print("API Key 长度:", len(api_key))
        print("Base URL:", base_url)
        
        # 创建一个没有代理的 httpx 异步客户端，避免 proxies 参数错误
        http_client = httpx.AsyncClient(trust_env=False)
        
        self.http_client = http_client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
        print(f"大模型客户端初始化完成 {self.client}")  
    
    async def close(self):
        """
        关闭异步客户端，释放资源
        """
        if self.http_client:
            await self.http_client.aclose()
            print("大模型客户端已关闭")
    async def generate_response_stream(self, context, query):
        """
        生成大模型回复（异步流式）
        
        Args:
            context: 上下文信息
            query: 用户查询
            
        Yields:
            str: 大模型回复的每一个片段
        """
        # 构建messages
        messages = [
            {"role": "system", "content": "你是一个专业的问答助手，请基于以下的上下文环境回答问题，只使用提供的文档内容回答，不要凭空编造信息。如果文档中没有相关信息，请回答'文档中没有相关信息'。"},
            {"role": "user", "content": f"上下文：{context}\n\n用户问题：{query}"}
        ]
        
        try:
            stream = await self.client.chat.completions.create(
                model=LLM_CONFIG["model_name"],
                messages=messages,
                stream=True
            )
            
            # 异步遍历输出项
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content
        except Exception as e:
            print(f"调用大模型失败: {e}")
            yield "生成回复失败"
