from zhipuai import ZhipuAI
from config import ZHIPU_CONFIG

class ZhipuClient:
    """
    智谱AI客户端类，使用zhipuai库
    """
    
    def __init__(self, api_key=None, base_url=None):
        """
        初始化智谱AI客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
        """
        self.api_key = api_key or ZHIPU_CONFIG.get("api_key", "")
        self.base_url = base_url or ZHIPU_CONFIG.get("base_url", "")
        
        self.client = ZhipuAI(api_key=self.api_key)
        print(f"智谱AI客户端初始化完成")
    
    def chat_completion(self, model, messages, max_tokens=500, temperature=0.7):
        """
        调用智谱AI聊天完成API
        
        Args:
            model: 使用的模型
            messages: 消息列表
            max_tokens: 最大输出tokens
            temperature: 控制输出的随机性
            
        Returns:
            str: 模型回复内容
        """
        try:
            print(f"发送请求到智谱AI: {self.base_url}")
            print(f"请求模型: {model}")
            
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            print(f"响应状态: 成功")
            
            # 获取完整回复
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"调用智谱AI失败: {e}")
            return ""