"""
工具函数模块
包含各种辅助函数
"""

import os
from datetime import datetime

def get_timestamp():
    """
    获取当前时间戳，用于文件命名
    
    返回:
        格式化的时间戳字符串 (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_unique_filename(base_path, type, timestamp=None):
    """
    创建带时间戳的唯一文件名
    
    参数:
        base_path: 基础文件路径（可能包含扩展名）
        timestamp: 时间戳，如果为None则使用当前时间
    
    返回:
        带时间戳的完整文件路径
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    # 分离目录、文件名和扩展名
    directory = os.path.dirname(base_path) # 获取目录部分，如果没有目录则为当前目录
    filename = os.path.basename(base_path) # 获取文件名部分，不包含目录
    
    # 如果base_path包含扩展名，分离出来
    if '.' in filename:
        name_part, ext_part = os.path.splitext(filename)
        timestamped_filename = f"{timestamp}_{name_part}_{type}{ext_part}"
    else:
        name_part = filename
        timestamped_filename = f"{timestamp}_{name_part}_{type}.csv"

    return os.path.join(directory, timestamped_filename)

