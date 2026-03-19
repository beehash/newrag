import os
import httpx
from openai import OpenAI

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
        
        # 创建一个没有代理的 httpx 客户端，避免 proxies 参数错误
        http_client = httpx.Client(trust_env=False)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
    
    def generate_response(self, context, query):
        """
        生成大模型回复
        
        Args:
            context: 上下文信息
            query: 用户查询
            
        Returns:
            str: 大模型回复
        """
        # 构建prompt
        prompt = f"""你是一个专业的问答助手，请基于以下的上下文环境回答问题：{context}，用户的问题：{query}
请注意：
1. 只使用提供的文档内容回答问题，不要凭空编造信息。
2. 如果文档中没有相关信息，请回答"文档中没有相关信息"。
3. 回答尽量简明扼要，逻辑清晰。
4. 如果涉及多个文档，请区分来源，并在必要时标注来源编号。

请开始回答："""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            # 遍历输出项
            final_answer = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    final_answer += chunk.choices[0].delta.content
            
            return final_answer if final_answer else "生成回复失败"
        except Exception as e:
            print(f"调用大模型失败: {e}")
            return "生成回复失败"
    
    def generate_response_stream(self, context, query):
        """
        生成大模型回复（流式）
        
        Args:
            context: 上下文信息
            query: 用户查询
            
        Yields:
            str: 大模型回复的每一个片段
        """
        # 构建prompt
        prompt = f"""你是一个专业的问答助手，请基于以下的上下文环境回答问题：{context}，用户的问题：{query}
请注意：
1. 只使用提供的文档内容回答问题，不要凭空编造信息。
2. 如果文档中没有相关信息，请回答"文档中没有相关信息"。
3. 回答尽量简明扼要，逻辑清晰。
4. 如果涉及多个文档，请区分来源，并在必要时标注来源编号。

请开始回答："""
        
        try:
            response = self.client.chat.completions.create(
                model="qwen3.5-plus",
                messages=[
                    {"role": "system", "content": "你是一个专业的问答助手"},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            
            # 遍历输出项
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"调用大模型失败: {e}")
            yield "生成回复失败"
