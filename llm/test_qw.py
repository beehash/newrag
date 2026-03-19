from openai import OpenAI
import httpx

# 直接在这里填入你的 API key 测试
api_key = "sk-54a9eaf0c047451d888a2387a1cd25d9"  # 替换为你的真实 API key
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

try:
    # 创建一个没有代理的 httpx 客户端
    http_client = httpx.Client(trust_env=False)
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=http_client
    )
    
    # 测试调用
    response = client.chat.completions.create(
        model="qwen3.5-plus",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ 连接成功!")
    print(response)
except Exception as e:
    print(f"❌ 错误: {e}")
