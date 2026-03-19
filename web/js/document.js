// 导入API函数
import { uploadFile, getDocs, getDoc, deleteDoc } from './api.js';

// DOM元素
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const fileList = document.getElementById('file-list');

// 页面加载时初始化
window.addEventListener('DOMContentLoaded', () => {
    // 加载文件列表
    loadFileList();
    
    // 绑定表单提交事件
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    // 绑定tab切换事件
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            switchTab(tabId, this);
        });
    });
});

/**
 * 切换Tab功能
 * @param {string} tabId - 要切换到的tab ID
 * @param {HTMLElement} btn - 点击的按钮元素
 */
function switchTab(tabId, btn) {
    // 隐藏所有tab内容
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // 移除所有tab按钮的active状态
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // 显示选中的tab内容
    const selectedTab = document.getElementById(tabId);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // 激活选中的tab按钮
    if (btn) {
        btn.classList.add('active');
    }
    
    // 如果切换到文件管理tab，重新加载文件列表
    if (tabId === 'files-tab') {
        loadFileList();
    }
}

/**
 * 处理文件上传
 * @param {Event} event - 表单提交事件
 */
async function handleFileUpload(event) {
    event.preventDefault();
    
    const files = fileInput.files;
    if (files.length === 0) {
        showStatus('请选择要上传的文件', 'error');
        return;
    }
    
    // 只处理第一个文件
    const file = files[0];
    
    // 创建FormData对象
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        showStatus('正在上传...', 'info');
        
        // 调用上传API
        const response = await uploadFile(formData);
        
        showStatus('上传成功！', 'success');
        
        // 清空文件输入
        fileInput.value = '';
        
        // 重新加载文件列表
        loadFileList();
    } catch (error) {
        showStatus(`上传失败: ${error.message}`, 'error');
    }
}

/**
 * 加载文件列表
 */
async function loadFileList() {
    try {
        const files = await getDocs();
        
        // 清空文件列表
        fileList.innerHTML = '';
        
        if (files.length === 0) {
            fileList.innerHTML = '<p class="empty-message">暂无文件</p>';
            return;
        }
        
        // 生成文件列表
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            const fileInfo = document.createElement('div');
            fileInfo.className = 'file-info';
            
            const fileName = document.createElement('div');
            fileName.className = 'file-name';
            fileName.textContent = file.filename;
            
            const fileMeta = document.createElement('div');
            fileMeta.className = 'file-meta';
            fileMeta.textContent = `类型: ${file.filetype} | 上传时间: ${new Date(file.create_at).toLocaleString()}`;
            
            fileInfo.appendChild(fileName);
            fileInfo.appendChild(fileMeta);
            
            const fileActions = document.createElement('div');
            fileActions.className = 'file-actions';
            
            const previewBtn = document.createElement('button');
            previewBtn.className = 'btn btn-secondary';
            previewBtn.textContent = '预览';
            previewBtn.addEventListener('click', () => handlePreviewFile(file.doc_id));
            
            const deleteBtn = document.createElement('button');
            deleteBtn.className = 'btn btn-danger';
            deleteBtn.textContent = '删除';
            deleteBtn.addEventListener('click', () => handleDeleteFile(file.doc_id));
            
            fileActions.appendChild(previewBtn);
            fileActions.appendChild(deleteBtn);
            
            fileItem.appendChild(fileInfo);
            fileItem.appendChild(fileActions);
            
            fileList.appendChild(fileItem);
        });
    } catch (error) {
        console.error('加载文件列表失败:', error);
        fileList.innerHTML = '<p class="empty-message">加载文件列表失败</p>';
    }
}

/**
 * 处理文件删除
 * @param {string} docId - 文档ID
 */
async function handleDeleteFile(docId) {
    if (!confirm(`确定要删除文件吗？`)) {
        return;
    }
    
    try {
        await deleteDoc(docId);
        loadFileList();
    } catch (error) {
        alert(`删除文件失败: ${error.message}`);
    }
}

/**
 * 处理文件预览
 * @param {string} docId - 文档ID
 */
async function handlePreviewFile(docId) {
    try {
        const fileContent = await getDoc(docId);
        
        // 创建预览模态框
        const modal = document.createElement('div');
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>文件预览</h3>
                    <button class="close-btn">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="markdown-content">${fileContent.content}</div>
                </div>
            </div>
        `;
        
        // 添加到页面
        document.body.appendChild(modal);
        
        // 显示模态框
        modal.style.display = 'block';
        
        // 关闭按钮事件
        const closeBtn = modal.querySelector('.close-btn');
        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
            document.body.removeChild(modal);
        });
        
        // 点击模态框外部关闭
        window.addEventListener('click', (event) => {
            if (event.target === modal) {
                modal.style.display = 'none';
                document.body.removeChild(modal);
            }
        });
    } catch (error) {
        alert(`预览文件失败: ${error.message}`);
    }
}

/**
 * 显示状态消息
 * @param {string} message - 消息内容
 * @param {string} type - 消息类型: success, error, info
 */
function showStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = `status-message ${type}`;
    
    // 3秒后清除消息
    setTimeout(() => {
        uploadStatus.textContent = '';
        uploadStatus.className = 'status-message';
    }, 3000);
}