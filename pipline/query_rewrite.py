import json
import requests

class QueryRewriter:
    """
    查询改写类，使用智谱AI的ChatGLM3-6B模型生成多个查询语句
    """
    
    def __init__(self, api_key="", base_url="https://open.bigmodel.cn/api/paas/v4"):
        """
        初始化查询改写器
        
        Args:
            api_key: 智谱AI API密钥
            base_url: 智谱AI API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def rewrite_query(self, query, model="glm-3-flash", num_rewrites=3):
        """
        改写查询语句
        
        Args:
            query: 原始查询语句
            model: 使用的模型
            num_rewrites: 生成的改写数量
            
        Returns:
            list: 改写后的查询语句列表
        """
        # 构建提示词
        prompt = f"""你是一个搜索优化助手。

请根据用户问题生成{num_rewrites}个更适合知识库检索的问题。

要求：
- 保持原意
- 使用不同表达方式
- 输出 JSON list 格式，例如：["问题1", "问题2", "问题3"]

用户问题：
{query}
"""
        
        # 发送请求到智谱AI
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            print(f"发送请求到智谱AI: {self.base_url}/chat/completions")
            print(f"请求模型: {model}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            print(f"响应状态码: {response.status_code}")
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 尝试解析JSON
            try:
                rewritten_queries = json.loads(content)
                
                # 确保返回的是列表
                if isinstance(rewritten_queries, list):
                    return rewritten_queries[:num_rewrites]
                else:
                    return [query]
            except json.JSONDecodeError:
                # 如果解析失败，返回原始查询
                print(f"JSON解析失败，返回原始查询: {content}")
                return [query]
        except Exception as e:
            print(f"查询改写失败: {e}")
            # 失败时返回原始查询
            return [query]
    
    def recognize_intent(self, query, model="glm-3-flash"):
        """
        识别查询意图
        
        Args:
            query: 原始查询语句
            model: 使用的模型
            
        Returns:
            dict: 意图识别结果
        """
        # 构建提示词
        prompt = f"""你是一个意图识别助手。

请分析用户问题的意图，并返回以下信息：
- intent: 意图类别（如：信息查询、技术支持、业务咨询等）
- confidence: 置信度（0-1之间的小数）
- keywords: 关键词列表
- summary: 问题摘要

要求：
- 输出 JSON 格式
- 意图类别要简洁明了
- 置信度要合理
- 关键词要准确反映问题核心
- 摘要要简洁概括问题

用户问题：
{query}
"""
        
        # 发送请求到智谱AI
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500
        }
        
        try:
            print(f"发送请求到智谱AI: {self.base_url}/chat/completions")
            print(f"请求模型: {model}")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            print(f"响应状态码: {response.status_code}")
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # 尝试解析JSON
            try:
                intent_result = json.loads(content)
                
                # 确保返回的是字典
                if isinstance(intent_result, dict):
                    return intent_result
                else:
                    return {
                        "intent": "其他",
                        "confidence": 0.5,
                        "keywords": [query],
                        "summary": query
                    }
            except json.JSONDecodeError:
                # 如果解析失败，返回默认值
                print(f"JSON解析失败，返回默认值: {content}")
                return {
                    "intent": "其他",
                    "confidence": 0.5,
                    "keywords": [query],
                    "summary": query
                }
        except Exception as e:
            print(f"意图识别失败: {e}")
            # 失败时返回默认值
            return {
                "intent": "其他",
                "confidence": 0.5,
                "keywords": [query],
                "summary": query
            }

# 测试代码
if __name__ == "__main__":
    rewriter = QueryRewriter()
    test_query = "如何使用Python实现文件上传功能？"
    
    # 测试查询改写
    rewritten_queries = rewriter.rewrite_query(test_query)
    print("原始查询:", test_query)
    print("改写查询:", rewritten_queries)
    
    # 测试意图识别
    intent_result = rewriter.recognize_intent(test_query)
    print("意图识别结果:", intent_result)
