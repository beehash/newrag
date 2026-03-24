// API 基础URL
const API_BASE_URL = 'http://127.0.0.1:8083/api';

// DOM元素
const chatForm = document.getElementById('chat-form');
const queryInput = document.getElementById('query-input');
const topkSlider = document.getElementById('topk-slider');
const topkValue = document.getElementById('topk-value');
const chatResult = document.getElementById('chat-result');

// 页面加载时初始化
window.addEventListener('DOMContentLoaded', () => {
    // 绑定slidebar变化事件
    topkSlider.addEventListener('input', (event) => {
        topkValue.textContent = event.target.value;
    });
    
    // 绑定表单提交事件
    if (chatForm) {
        chatForm.addEventListener('submit', handleQuery);
    }
});

/**
 * 处理查询请求
 * @param {Event} event - 表单提交事件
 */
async function handleQuery(event) {
    event.preventDefault();
    
    const query = queryInput.value.trim();
    if (!query) {
        alert('请输入查询内容');
        return;
    }
    
    const topk = parseInt(topkSlider.value);
    
    try {
        // 显示加载状态
        chatResult.innerHTML = '<p>正在搜索...</p>';
        
        // 发送查询请求
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                q: query,
                topk: topk
            })
        });
        
        if (!response.ok) {
            throw new Error(`查询失败: ${response.statusText}`);
        }
        
        // 处理流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // 清空结果区域
        chatResult.innerHTML = '';
        
        // 存储意图和资源数据
        let intentData = null;
        let resourcesData = null;
        
        // 存储回答内容
        let answerContent = '';
        let answerElement = null;
        
        // 读取流数据
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // 解码数据
            const chunk = decoder.decode(value, { stream: true });
            
            const events = chunk.split('\n\n');
            for (const event of events) {
                if (!event || !event.startsWith('data: ')) continue;
                
                const dataStr = event.replace('data: ', '');
                try {
                    const data = JSON.parse(dataStr);
                    
                    if (data.type === 'init') {
                        // 处理初始化数据
                        intentData = data.intent;
                        resourcesData = data.resources;
                        
                        // 显示意图识别结果
                        if (intentData) {
                            const intentSection = document.createElement('div');
                            intentSection.className = 'intent-section';
                            intentSection.innerHTML = `
                                <h3>意图分析</h3>
                                <p><strong>意图类别:</strong> ${intentData.intent || '未知'}</p>
                                <p><strong>置信度:</strong> ${Math.round((intentData.confidence || 0) * 100)}%</p>
                                <p><strong>实体词:</strong> ${(intentData.entities || []).join(', ')}</p>
                                <p><strong>问题摘要:</strong> ${intentData.summary || ''}</p>
                            `;
                            chatResult.appendChild(intentSection);
                        }
                        
                        // 显示大模型输出结果容器
                        const resultContent = document.createElement('div');
                        resultContent.className = 'result-content';
                        resultContent.innerHTML = '<h3>回答</h3><div class="markdown-content"></div>';
                        chatResult.appendChild(resultContent);
                        answerElement = resultContent.querySelector('.markdown-content');
                        
                        // 显示资源列表
                        if (resourcesData && resourcesData.length > 0) {
                            const resourcesSection = document.createElement('div');
                            resourcesSection.className = 'result-resources';
                            resourcesSection.innerHTML = '<h3>相关资源</h3>';
                            
                            resourcesData.forEach(resource => {
                                const resourceItem = document.createElement('div');
                                resourceItem.className = 'resource-item';
                                
                                const resourceTitle = document.createElement('div');
                                resourceTitle.className = 'resource-title';
                                resourceTitle.textContent = resource.title || '无标题';
                                
                                const resourceContent = document.createElement('div');
                                resourceContent.className = 'resource-content';
                                resourceContent.textContent = resource.text || '无内容';
                                
                                const resourceSimilarity = document.createElement('div');
                                resourceSimilarity.className = 'resource-similarity';
                                let similarity = 0;
                                if (resource.similarity) {
                                    similarity = Math.round((resource.similarity || 0) * 100) + '%';
                                } else {
                                    similarity = resource._similarity || 0;
                                }
                                resourceSimilarity.textContent = `匹配度: ${similarity}`;
                                
                                resourceItem.appendChild(resourceTitle);
                                resourceItem.appendChild(resourceContent);
                                resourceItem.appendChild(resourceSimilarity);
                                resourcesSection.appendChild(resourceItem);
                            });
                            
                            chatResult.appendChild(resourcesSection);
                        } else {
                            const noResources = document.createElement('div');
                            noResources.className = 'result-resources';
                            noResources.innerHTML = '<h3>相关资源</h3><p>暂无相关资源</p>';
                            chatResult.appendChild(noResources);
                        }
                    } else if (data.type === 'chunk' && answerElement) {
                        // 处理回答片段
                        answerContent += data.content;
                        // 使用 Markdown
                        async () => { answerElement.innerHTML = await marked.parse(answerContent) };   
                        const asyncParse = new Promise((resolve) => {
                            resolve(marked.parse(answerContent));
                        });
                        
                        answerElement.innerHTML = await asyncParse;
                    } else if (data.type === 'end') {
                        // 处理结束信号
                        console.log('流式响应结束');
                    }
                } catch (e) {
                    console.error('解析流式数据失败:', e);
                }
            }
        }
    } catch (error) {
        console.error('查询时出错:', error);
        chatResult.innerHTML = `<p class="error-message">查询失败: ${error.message}</p>`;
    }
}

/**
 * 显示查询结果
 * @param {Object} data - 查询结果数据
 */
function displayResult(data) {
    // 清空结果区域
    chatResult.innerHTML = '';
    
    // 显示意图识别结果
    if (data.intent) {
        const intentSection = document.createElement('div');
        intentSection.className = 'intent-section';
        intentSection.innerHTML = `
            <h3>意图分析</h3>
            <p><strong>意图类别:</strong> ${data.intent.intent || '未知'}</p>
            <p><strong>置信度:</strong> ${Math.round((data.intent.confidence || 0) * 100)}%</p>
            <p><strong>关键词:</strong> ${(data.intent.entities || []).join(', ')}</p>
            <p><strong>问题摘要:</strong> ${data.intent.summary || ''}</p>
        `;
        chatResult.appendChild(intentSection);
    }
    
    // 显示大模型输出结果
    const resultContent = document.createElement('div');
    resultContent.className = 'result-content';
    
    // 使用 marked 库渲染 Markdown 格式的回答
    const answerHtml = marked.parse(data.answer || '暂无回答');
    resultContent.innerHTML = `<h3>回答</h3><div class="markdown-content">${answerHtml}</div>`;
    chatResult.appendChild(resultContent);
    
    // 显示资源列表
    if (data.resources && data.resources.length > 0) {
        const resourcesSection = document.createElement('div');
        resourcesSection.className = 'result-resources';
        resourcesSection.innerHTML = '<h3>相关资源</h3>';
        
        data.resources.forEach(resource => {
            const resourceItem = document.createElement('div');
            resourceItem.className = 'resource-item';
            
            const resourceTitle = document.createElement('div');
            resourceTitle.className = 'resource-title';
            resourceTitle.textContent = resource.title || '无标题';
            
            const resourceContent = document.createElement('div');
            resourceContent.className = 'resource-content';
            resourceContent.textContent = resource.content || '无内容';
            
            const resourceSimilarity = document.createElement('div');
            resourceSimilarity.className = 'resource-similarity';
            resourceSimilarity.textContent = '';//`相似度: ${Math.round((resource.similarity || 0) * 100)}%`;
            
            resourceItem.appendChild(resourceTitle);
            resourceItem.appendChild(resourceContent);
            resourceItem.appendChild(resourceSimilarity);
            resourcesSection.appendChild(resourceItem);
        });
        
        chatResult.appendChild(resourcesSection);
    } else {
        const noResources = document.createElement('div');
        noResources.className = 'result-resources';
        noResources.innerHTML = '<h3>相关资源</h3><p>暂无相关资源</p>';
        chatResult.appendChild(noResources);
    }
}
