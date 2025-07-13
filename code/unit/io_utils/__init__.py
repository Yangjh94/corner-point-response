"""
SAP2000风振分析包

此包提供了SAP2000模型操作和风振分析的完整功能
"""

from .sap_model import SAP2000Model
from .wind_manager import WindAnalysisManager
from .utils import get_timestamp, create_unique_filename

__version__ = "1.0.0"
__author__ = "Your Name"

# 导出主要类和函数
__all__ = [
    'SAP2000Model',
    'WindAnalysisManager', 
    'get_timestamp',
    'create_unique_filename'
]