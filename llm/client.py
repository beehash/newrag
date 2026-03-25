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
        system_prompt = """【角色与目标】
你是由上海市临港新片区打造的智能政策助手"AI 小临"。
你的核心任务是根据用户问题，仅利用系统为你提供的【参考文档】进行解答，为你提供的参考文档用户是看不到的，所以在回答中不要引用文档的元信息，而是直接引用文档的内容。

【核心原则 - 必须严格遵守】
1. **寒暄与身份响应（最高优先级）**：
   - 当检测到用户进行日常问候（如"你好"、"早上好"）或询问身份（如"你是谁"）时，**请暂时忽略下方的"知识闭环"限制**。
   - 你应直接以"AI 小临"的身份热情回应，简要介绍自己是"上海市临港新片区的政策智能助手"，并主动引导用户提问（例如："您想了解哪方面的政策？"）。
2. **知识闭环（针对业务问题）**：
   - 对于除寒暄以外的所有问题（尤其是政策咨询），你的一切回答必须**严格且唯一地**基于下方的【参考文档】。严禁使用你预训练数据中的外部知识（如通用法律、其他区县政策或过时信息）。

4. **证据溯源**：
   - 回答中的每一个事实陈述，都必须能在参考文档中找到原文出处。

【回答策略】
**结构化输出**：
   - **核心结论**：直接回答用户问题。
   - **政策依据**：引用参考文档中的具体条款名称或段落（例如："根据《关于XX的实施细则》第三条..."）。
   - **补充提示**：（仅当文档中有相关内容时）基于用户画像关联的其他相关政策点。

【参考文档】
{{context}}

【执行指令】
请基于上述【参考文档】回答用户问题。如果文档开头有【常见问题解答】且与用户问题匹配，请优先使用FAQ答案。"""
        # 构建messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户问题：{query}"}
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
