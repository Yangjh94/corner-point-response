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
        type: 文件类型标识
        timestamp: 时间戳，如果为None则使用当前时间
    
    返回:
        带时间戳的完整文件路径
    """
    if timestamp is None:
        timestamp = get_timestamp()
    
    # 分离目录、文件名和扩展名
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    
    # 如果base_path包含扩展名，分离出来
    if '.' in filename:
        name_part, ext_part = os.path.splitext(filename)
    else:
        name_part = filename
        ext_part = ""

    # 创建带时间戳的文件名
    timestamped_filename = f"{timestamp}_{name_part}_{type}{ext_part}"

    return os.path.join(directory, timestamped_filename)

def validate_file_exists(file_path):
    """
    验证文件是否存在
    
    参数:
        file_path: 文件路径
    
    返回:
        bool: 文件是否存在
    """
    return os.path.exists(file_path)

def ensure_directory_exists(directory_path):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory_path: 目录路径
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"创建目录: {directory_path}")