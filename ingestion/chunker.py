def chunk_text(text, chunk_size=1000):
    """
    将文本按照指定大小切块，当接近chunk_size时，选择最靠近的句号切断
    
    Args:
        text: 要切块的文本
        chunk_size: 切块大小，默认为1000字符
    
    Returns:
        切块后的文本列表
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # 计算当前块的结束位置
        end = start + chunk_size
        
        # 如果已经到达文本末尾，直接添加剩余部分
        if end >= text_length:
            chunks.append(text[start:])
            break
        
        # 寻找最靠近end的句号
        # 从end向前搜索，最多搜索200个字符
        search_start = max(start, end - 200)
        search_text = text[search_start:end]
        
        # 查找句号位置
        period_pos = max(
            search_text.rfind('。'),  # 中文句号
            search_text.rfind('.')    # 英文句号
        )
        
        # 如果找到句号，调整end位置
        if period_pos != -1:
            end = search_start + period_pos + 1
        
        # 添加当前块
        chunks.append(text[start:end])
        
        # 更新start位置
        start = end
    
    return chunks