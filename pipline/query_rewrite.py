import json
import requests
import re

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
    
    def rewrite_query(self, query, model="glm-4-flash", num_rewrites=3):
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
        prompt = f"""请将用户问题改写为更适合知识库检索的查询：

请根据用户问题生成{num_rewrites}个更适合知识库检索的问题。

要求：
1. 保留核心实体（必须保留原词）
2. 从两个维度扩展：
   - 意图（流程、条件、材料等）
   - 背景（政策、类别、同义概念）
3. 每个扩展都必须包含核心实体
4. 输出不超过5个查询
5. 去掉无意义动词（如：怎么、如何、申请）
6. 可以扩展同义表达，保持查询的多样性，核心实体不能丢失
7. 输出 JSON list 格式，例如：["问题1", "问题2", "问题3"]

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
            print(f"响应内容1: {response}")
            response.raise_for_status()
            print(f"响应内容2: {response}")
            # 解析响应
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"原始响应内容: {content}")
            
            # 尝试解析JSON
            try:
                content = content.strip()
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
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
    
    def recognize_intent(self, query, model="glm-4-flash"):
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
                # 移除 ```json 或 ``` 标记
                content = content.strip()
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
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

 
    def analyze_query(self, query, model="glm-4-flash", num_rewrites=3):
        """
        分析查询：同时进行意图识别和查询改写
        
        Args:
            query: 原始查询语句
            model: 使用的模型
            num_rewrites: 生成的改写数量
            
        Returns:
            dict: 包含意图识别结果和改写查询的字典
        """
        # 构建综合提示词
        prompt = f"""你是一个智能查询分析助手。
请对用户问题进行综合分析：

1. 意图识别：
   - intent: 意图类别（如：信息查询、技术支持、业务咨询等）
   - confidence: 置信度（0-1之间的小数）
   - summary: 问题摘要
   - entities: 核心实体列表，每个实体只写入实体名称，如：["居住证", "人才卡"]

2. 查询改写：
   - 生成{num_rewrites}个更适合知识库检索的问题并用列表返回
   - 保持原意图实体不变
   - 根据实体词扩展查询范围, 实体词重复2次增加实体词的权重, 保持查询的连贯性

要求：
- 输出 JSON 格式，包含两个字段："intent" 和 "rewritten_queries"
- "intent" 是一个包含意图识别结果的字典
- "rewritten_queries" 是一个包含改写查询的列表

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
            "max_tokens": 1000
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
            print(f"模型回复(综合分析): {content}")
            
            # 尝试解析JSON
            try:
                # 移除 ```json 或 ``` 标记
                content = content.strip()
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'^```\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                content = content.strip()
                
                # 检查内容是否为空
                if not content:
                    print("内容为空，返回默认值")
                    return {
                        "intent": {
                            "intent": "其他",
                            "confidence": 0.5,
                            "entities": [query],
                            "summary": query
                        },
                        "rewritten_queries": [query]
                    }
                
                # 尝试直接解析JSON
                try:
                    analysis_result = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON解析失败: {str(e)}")
                    # 如果解析失败，返回默认值
                    return {
                        "intent": {
                            "intent": "其他",
                            "confidence": 0.5,
                            "entities": [query],
                            "summary": query
                        },
                        "rewritten_queries": [query]
                    }
                
                # 确保返回的是包含正确字段的字典
                if isinstance(analysis_result, dict):
                    # 提取意图识别结果
                    intent_result = analysis_result.get("intent", {
                        "intent": "其他",
                        "confidence": 0.5,
                        "entities": [query],
                        "summary": query
                    })
                
                    
                    # 提取改写查询
                    rewritten_queries = analysis_result.get("rewritten_queries", [query])
                    if not isinstance(rewritten_queries, list):
                        rewritten_queries = [query]
                    
                    return {
                        "intent": intent_result,
                        "rewritten_queries": rewritten_queries[:num_rewrites]
                    }
                else:
                    print("解析结果不是字典，返回默认值")
                    return {
                        "intent": {
                            "intent": "其他",
                            "confidence": 0.5,
                            "entities": [query],
                            "summary": query
                        },
                        "rewritten_queries": [query]
                    }
            except json.JSONDecodeError:
                # 如果解析失败，返回默认值
                print(f"JSON解析失败，返回默认值: {content}")
                return {
                    "intent": {
                        "intent": "其他",
                        "confidence": 0.5,
                        "entities": [query],
                        "summary": query
                    },
                    "rewritten_queries": [query]
                }
        except Exception as e:
            print(f"查询分析失败: {e}")
            # 失败时返回默认值
            return {
                "intent": {
                    "intent": "其他",
                    "confidence": 0.5,
                    "entities": [query],
                    "summary": query
                },
                "rewritten_queries": [query]
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
