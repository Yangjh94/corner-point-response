"""
主程序入口
风振分析程序的主要执行文件
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))
from utils.io_utils.wind_manager import WindAnalysisManager

def main():
    """主程序 - 简化版"""
    # 创建分析管理器
    analyzer = WindAnalysisManager(output_dir="output")
    
    # 设置参数
    wind_files = ["Model2_10yr_000.csv"]  # 您的风荷载文件列表
    
    target_elevations = [6000, 10500, 15000, 19500, 23100, 26700, 30300, 33900, 37500, 41100, 
                         44700, 48300, 51900, 55500, 59100, 62700, 66300, 69900, 73500, 77100, 
                         80700, 84300, 87900, 91500, 95100, 98700, 102300, 105900, 109500, 113100, 
                         116700, 120300, 123900, 127500, 131100, 134700, 138300, 141900, 145500, 
                         149100, 152700, 156300, 159900, 163500, 167100, 170700, 174300, 177900, 
                         181500, 185100, 188700, 192300, 196150, 200000]
    
    target_nodes = ["54000062", "54000070", "54000071", "54000079"]
    
    # 运行批量分析
    success = analyzer.run_batch_analysis(
        wind_files, 
        target_elevations, 
        target_nodes,
        num_rows=33,
        tolerance=10,
        load_case='Wind_time_history'
    )
    
    if success:
        print("批量分析完成！")
        # 输出统计结果
        analyzer.summarize_results()
    else:
        print("批量分析失败！")

if __name__ == "__main__":
    main()