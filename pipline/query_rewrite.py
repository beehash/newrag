import json
import requests

class QueryRewriter:
    """
    查询改写类，使用ollama-qwen3:8b模型生成多个查询语句
    """
    
    def __init__(self, ollama_url="http://localhost:11434/api/generate"):
        """
        初始化查询改写器
        
        Args:
            ollama_url: ollama API地址
        """
        self.ollama_url = ollama_url
    
    def rewrite_query(self, query, model="qwen2.5:7b", num_rewrites=3):
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
- 输出 JSON list

用户问题：
{query}
"""
        
        # 发送请求到ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        
        try:
            print(f"发送请求到Ollama: {self.ollama_url}")
            print(f"请求参数: {payload}")
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            rewritten_queries = json.loads(result.get("response", "[]"))
            
            # 确保返回的是列表且长度正确
            if isinstance(rewritten_queries, dict) and "questions" in rewritten_queries:
                # 如果是字典且包含 questions 字段，返回 questions 列表
                if isinstance(rewritten_queries["questions"], list):
                    return rewritten_queries["questions"][:num_rewrites]
                else:
                    return [query]
            elif isinstance(rewritten_queries, list):
                # 如果是列表，直接返回
                return rewritten_queries[:num_rewrites]
            else:
                return [query]
        except Exception as e:
            print(f"查询改写失败: {e}")
            # 失败时返回原始查询
            return [query]
    
    def recognize_intent(self, query, model="qwen2.5:7b"):
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
        
        # 发送请求到ollama
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False
        }
        
        try:
            print(f"发送请求到Ollama: {self.ollama_url}")
            print(f"请求参数: {payload}")
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            intent_result = json.loads(result.get("response", "{}"))
            
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

