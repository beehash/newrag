// API 基础URL
const API_BASE_URL = 'http://127.0.0.1:8083/api';

/**
 * 上传文件到服务器
 * @param {FormData} formData - 包含文件的FormData对象
 * @returns {Promise<Object>} - 服务器返回的响应
 */
export async function uploadFile(formData) {
    try {
        console.log('开始上传文件...');
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        console.log('上传请求发送完成',response);
        if (!response.ok) {
            throw new Error(`上传失败: ${response.statusText}`);
        }
        console.log('上传成功:', response)
        return await response.json();
    } catch (error) {
        console.error('上传文件时出错:', error);
        throw error;
    }
}

/**
 * 获取所有文件列表
 * @returns {Promise<Array>} - 文件列表
 */
export async function getDocs() {
    try {
        const response = await fetch(`${API_BASE_URL}/docs`);
        
        if (!response.ok) {
            throw new Error(`获取文件列表失败: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('获取文件列表时出错:', error);
        throw error;
    }
}

/**
 * 获取单个文件内容
 * @param {string} docId - 文档ID
 * @returns {Promise<Object>} - 文件内容
 */
export async function getDoc(docId) {
    try {
        const response = await fetch(`${API_BASE_URL}/doc/${docId}`);
        
        if (!response.ok) {
            throw new Error(`获取文件内容失败: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('获取文件内容时出错:', error);
        throw error;
    }
}

/**
 * 删除文件
 * @param {string} docId - 文档ID
 * @returns {Promise<Object>} - 服务器返回的响应
 */
export async function deleteDoc(docId) {
    try {
        const response = await fetch(`${API_BASE_URL}/doc/${docId}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`删除文件失败: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('删除文件时出错:', error);
        throw error;
    }
}