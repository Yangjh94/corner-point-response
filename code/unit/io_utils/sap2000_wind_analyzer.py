# sap2000_wind_analyzer.py
import os
import pandas as pd
import comtypes.client
from datetime import datetime
from collections import defaultdict

class SAP2000Model:
    """SAP2000模型管理器 - 核心类"""
    def __init__(self):
        self.model = None
        self.is_connected = False
    
    def connect(self):
        """连接到SAP2000"""
        try:
            helper = comtypes.client.CreateObject('SAP2000v1.Helper')
            helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
            sap_object = helper.GetObject("CSI.SAP2000.API.SapObject")
            self.model = sap_object.SapModel
            self.is_connected = True
            
            # 解锁模型
            if self.model.GetModelIsLocked():
                self.model.SetModelIsLocked(False)
            
            print("成功连接到SAP2000")
            return True
        except Exception as e:
            print(f"连接SAP2000失败: {e}")
            return False
    
    def add_diaphragms(self, target_elevations, tolerance=10):
        """添加刚性隔板 - 直接集成到模型类中"""
        # 您的原始 add_diaphragms 代码
        node_info = []
        node_z_coords = defaultdict(list)
        constraint_names = []

        number_of_points, point_names, ret = self.model.PointObj.GetNameList()
        
        for point_name in point_names:
            [x, y, z, ret] = self.model.PointObj.GetCoordCartesian(point_name)
            node_info.append({"name": point_name, "x": x, "y": y, "z": z})
        
        # 其余代码保持不变...
        return constraint_names, node_z_coords
    
    def add_wind_loads(self, wind_file_path, diaphragm_constraints, node_z_coords, num_rows=None):
        """添加风荷载 - 直接集成到模型类中"""
        # 您的原始 add_wind_time_history_load 代码
        # 保持不变...
        pass
    
    def setup_modal_analysis(self, case_name="Wind_time_history", num_modes=30, damping=0.02):
        """设置模态分析"""
        # 设置模态分析工况
        ret = self.model.LoadCases.Delete("MODAL")
        ret = self.model.LoadCases.ModalEigen.SetCase("MODAL")
        ret = self.model.LoadCases.ModalEigen.SetNumberModes("MODAL", num_modes, 1)
        
        # 设置模态时程分析
        ret = self.model.LoadCases.Delete(case_name)
        ret = self.model.LoadCases.ModHistLinear.SetCase(case_name)
        ret = self.model.LoadCases.ModHistLinear.SetModalCase(case_name, "MODAL")
        ret = self.model.LoadCases.ModHistLinear.SetDampConstant(case_name, damping)
        
        return ret == 0
    
    def run_analysis(self):
        """运行分析"""
        ret = self.model.Analyze.SetSolverOption_1(2, 0, True)  # 多线程
        ret = self.model.Analyze.RunAnalysis()
        return ret == 0
    
    def get_node_response(self, node_name, load_case="Wind_time_history"):
        """获取节点响应 - 直接集成到模型类中"""
        # 您的原始 get_node_response_history 代码
        # 保持不变...
        pass

class WindAnalysisManager:
    """风振分析管理器 - 只负责批量处理和结果管理"""
    def __init__(self, output_dir="output"):
        self.sap_model = SAP2000Model()
        self.output_dir = output_dir
        self.results = []
    
    def run_batch_analysis(self, wind_files, target_elevations, target_nodes, **params):
        """批量运行风振分析"""
        if not self.sap_model.connect():
            return False
        
        # 创建刚性隔板（只需要做一次）
        diaphragm_constraints, node_z_coords = self.sap_model.add_diaphragms(
            target_elevations, params.get('tolerance', 10)
        )
        
        for wind_file in wind_files:
            print(f"处理风荷载文件: {wind_file}")
            
            # 添加风荷载
            wind_load_count, diaphragm_centers = self.sap_model.add_wind_loads(
                wind_file, diaphragm_constraints, node_z_coords, 
                params.get('num_rows')
            )
            
            # 设置分析工况
            self.sap_model.setup_modal_analysis(
                params.get('case_name', 'Wind_time_history'),
                params.get('num_modes', 30),
                params.get('damping', 0.02)
            )
            
            # 运行分析
            if self.sap_model.run_analysis():
                # 提取结果
                for node in target_nodes:
                    times, displacements, accelerations = self.sap_model.get_node_response(
                        node, params.get('case_name', 'Wind_time_history')
                    )
                    
                    # 保存结果
                    self._save_results(wind_file, node, times, displacements, accelerations)
                    
                    # 存储到内存
                    self.results.append({
                        "wind_file": wind_file,
                        "node": node,
                        "times": times,
                        "displacements": displacements,
                        "accelerations": accelerations
                    })
    
    def _save_results(self, wind_file, node, times, displacements, accelerations):
        """保存结果到文件"""
        wind_name = os.path.splitext(os.path.basename(wind_file))[0]
        results_dir = os.path.join(self.output_dir, "Timehistory", wind_name)
        os.makedirs(results_dir, exist_ok=True)