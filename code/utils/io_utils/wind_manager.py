"""
风振分析管理器
负责批量处理和结果管理
"""

import os
import time
from datetime import datetime
from .sap_model import SAP2000Model
from .utils import create_unique_filename

class WindAnalysisManager:
    """风振分析管理器 - 负责批量处理和结果管理"""
    
    def __init__(self, output_dir="output"):
        self.sap_model = SAP2000Model()
        self.output_dir = output_dir
        self.all_results = []
    
    def run_batch_analysis(self, wind_files, target_elevations, target_nodes, **params):
        """批量运行风振分析 - 基于您的main函数逻辑"""
        start_time = time.time()
        start_datetime = datetime.now()
        
        print("=" * 80)
        print("SAP2000风振分析程序")
        print(f"程序开始时间: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # 连接SAP2000
        if not self.sap_model.connect():
            print("无法连接到SAP2000，程序终止")
            return False
        
        # 创建刚性隔板（只需要做一次）
        print("\n[步骤1] 创建刚性隔板...")
        diaphragm_constraints, node_z_coords = self.sap_model.add_diaphragms(
            target_elevations, params.get('tolerance', 10)
        )
        
        if not diaphragm_constraints:
            print("创建刚性隔板失败")
            return False
        
        # 循环处理每个风荷载文件
        for wind_file_name in wind_files:
            print(f"\n[步骤2] 处理风荷载文件: {wind_file_name}")
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            wind_file_path = os.path.join(os.path.dirname(script_dir), "WindloadTimes", wind_file_name)
            
            # 添加风荷载
            wind_load_count, diaphragm_centers = self.sap_model.add_wind_time_history_load(
                diaphragm_constraints, 
                node_z_coords, 
                wind_file_path, 
                params.get('num_rows', 33)
            )
            
            if wind_load_count <= 0:
                print(f"添加风荷载失败: {wind_file_name}")
                continue
            
            # 运行分析
            print(f"\n[步骤3] 运行分析: {wind_file_name}")
            ret = self.sap_model.model.Analyze.SetSolverOption_1(2, 0, True)
            ret = self.sap_model.model.Analyze.RunAnalysis()
            
            if ret != 0:
                print(f"分析失败: {wind_file_name}")
                continue
            
            print("分析成功完成")
            
            # 提取结果
            self._extract_and_save_results(
                wind_file_name, wind_file_path, target_nodes, 
                diaphragm_centers, params
            )
        
        # 计算耗时并输出统计
        self._print_summary(wind_files, start_time, start_datetime)
        return True
    
    def _extract_and_save_results(self, wind_file_name, wind_file_path, target_nodes, diaphragm_centers, params):
        """提取并保存结果"""
        # 获取最高楼层节点
        top_diaphragm_center_name = max(diaphragm_centers.keys(), 
                                       key=lambda x: float(x.split('_')[-1]))
        node_top_center_name = diaphragm_centers[top_diaphragm_center_name]["point_name"]
        
        # 创建包含顶层节点的完整节点列表
        all_target_nodes = [node_top_center_name] + target_nodes
        
        # 创建结果目录
        wind_file_base_name = os.path.splitext(wind_file_name)[0]
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(script_dir, self.output_dir, "Timehistory", wind_file_base_name)
        os.makedirs(results_dir, exist_ok=True)
        
        # 为每个节点提取结果
        for target_node in all_target_nodes:
            print(f"\n获取节点 {target_node} 的响应...")
            
            output_path = os.path.join(results_dir, f"{target_node}.csv")
            
            times, displacements, accelerations = self.sap_model.get_node_response_history(
                target_node,
                load_case=params.get('load_case', 'Wind_time_history'),
                output_file=output_path
            )
            
            if times and displacements and accelerations:
                print(f"成功获取节点 {target_node} 的 {len(times)} 个时间步数据")
                
                # 存储结果
                self.all_results.append({
                    "wind_file": wind_file_name,
                    "node": target_node,
                    "times": times,
                    "displacements": displacements,
                    "accelerations": accelerations
                })
            else:
                print(f"获取节点 {target_node} 响应失败")
    
    def _print_summary(self, wind_files, start_time, start_datetime):
        """输出程序统计信息"""
        end_time = time.time()
        end_datetime = datetime.now()
        total_time = end_time - start_time
        
        print("=" * 80)
        print("程序执行完成")
        print(f"共运行了{len(wind_files)}个风荷载时程文件")
        print(f"程序结束时间: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"程序总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
        print("=" * 80)
    
    def get_results(self):
        """获取所有分析结果"""
        return self.all_results
    
    def summarize_results(self):
        """结果统计分析"""
        import pandas as pd
        
        # 提取统计信息
        summary_data = []
        for result in self.all_results:
            summary_data.append({
                "wind_file": result["wind_file"],
                "node": result["node"],
                "time_steps": len(result["times"]),
                "max_displacement": max(max(map(abs, result["displacements"][0]), default=0),
                                        max(map(abs, result["displacements"][1]), default=0),
                                        max(map(abs, result["displacements"][2]), default=0)),
                "max_acceleration": max(max(map(abs, result["accelerations"][0]), default=0),
                                        max(map(abs, result["accelerations"][1]), default=0),
                                        max(map(abs, result["accelerations"][2]), default=0))
            })

        # 转换为DataFrame并分组统计
        df_summary = pd.DataFrame(summary_data)
        grouped_summary = df_summary.groupby("wind_file").agg({
            "node": "count",
            "time_steps": "sum",
            "max_displacement": "max",
            "max_acceleration": "max"
        }).reset_index()

        print("\n统计结果表格:")
        print(grouped_summary)
        
        return grouped_summary