"""
简化版风振分析数据处理程序 - 专门处理Ux、Uy、Rz三个方向的响应数据
"""
# 导入所需的库
import os
import pandas as pd
import matplotlib.pyplot as plt

# 设置文件路径和目标列
output_folder = os.path.join(os.getcwd(), "data", "output", "Timehistory_modal", "1-1", "Acceleration")
target_columns = ['UX', 'UY', 'RZ']  # 目标分析列

# 获取所有文件夹名称
folder_names = os.listdir(output_folder)

# 存储结果的列表
results_data = []
angles = []
plot_data = {'mean': {col: [] for col in target_columns}, 
             'std': {col: [] for col in target_columns}, 
             'extreme': {col: [] for col in target_columns}}

# 处理每个文件夹（代表不同角度）
for folder_name in folder_names:
    print(f"\n处理文件夹: {folder_name}")
    
    # 提取角度信息
    try:
        angle = int(folder_name.split('_')[-1])
    except:
        angle = len(angles)
    angles.append(angle)
    
    # 获取文件夹下的CSV文件
    folder_path = os.path.join(output_folder, folder_name)
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    last_5_files = csv_files[-5:]  # 取最后5个文件
    
    # 存储当前文件夹的数据
    folder_stats = {col: {'means': [], 'stds': []} for col in target_columns}
    
    # 处理每个CSV文件
    for i, csv_file in enumerate(last_5_files, 1):
        print(f"  处理第{i}个文件: {csv_file}")
        
        # 读取数据并筛选目标列
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        
        # 检查目标列是否存在
        available_cols = [col for col in target_columns if col in data.columns]
        if not available_cols:
            print(f"    警告：文件中没有找到目标列，跳过")
            continue
        
        # 计算统计值并保存结果
        file_data = data[available_cols]
        for col in available_cols:
            mean_val = file_data[col].mean()
            std_val = file_data[col].std()
            extreme_val = mean_val + 3.5 * std_val
            
            # 保存到文件夹统计中
            folder_stats[col]['means'].append(mean_val)
            folder_stats[col]['stds'].append(std_val)
            
            # 保存详细结果
            results_data.append({
                'folder': folder_name,
                'angle': angle,
                'file': csv_file,
                'metric': col,
                'mean': round(mean_val, 2),
                'std': round(std_val, 2),
                'extreme': round(extreme_val, 2)
            })
    
    # 计算文件夹级别的平均值（用于绘图）
    for col in target_columns:
        if folder_stats[col]['means']:  # 如果有数据
            folder_mean_avg = sum(folder_stats[col]['means']) / len(folder_stats[col]['means'])
            folder_std_avg = sum(folder_stats[col]['stds']) / len(folder_stats[col]['stds'])
            folder_extreme_avg = folder_mean_avg + 3.5 * folder_std_avg
            
            plot_data['mean'][col].append(folder_mean_avg)
            plot_data['std'][col].append(folder_std_avg)
            plot_data['extreme'][col].append(folder_extreme_avg)
        else:
            # 如果没有数据，添加0值
            plot_data['mean'][col].append(0)
            plot_data['std'][col].append(0)
            plot_data['extreme'][col].append(0)

# 创建可视化图表
if results_data:
    print(f"\n开始绘图...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  
    plt.rcParams['axes.unicode_minus'] = False    
    
    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    
    # 绘制均值图
    for col in target_columns:
        if plot_data['mean'][col]:  # 如果有数据
            ax1.plot(angles, plot_data['mean'][col], marker='o', label=f'{col}_均值', linewidth=2)
    ax1.set_xlabel('角度 (度)')
    ax1.set_ylabel('均值')
    ax1.set_title('各角度下响应均值变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制标准差图
    for col in target_columns:
        if plot_data['std'][col]:
            ax2.plot(angles, plot_data['std'][col], marker='s', label=f'{col}_标准差', linewidth=2)
    ax2.set_xlabel('角度 (度)')
    ax2.set_ylabel('标准差')
    ax2.set_title('各角度下响应标准差变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 绘制极值图
    for col in target_columns:
        if plot_data['extreme'][col]:
            ax3.plot(angles, plot_data['extreme'][col], marker='^', label=f'{col}_极值', linewidth=2, markersize=8)
    ax3.set_xlabel('角度 (度)')
    ax3.set_ylabel('极值 (均值+3.5×标准差)')
    ax3.set_title('各角度下响应极值变化')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_parent = os.path.dirname(output_folder)
    plot_path = os.path.join(output_parent, "响应分析图.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存: {plot_path}")
    plt.show()

# 保存结果到CSV文件
if results_data:
    # 转换为DataFrame
    results_df = pd.DataFrame(results_data)
    
    print("\n结果预览:")
    print(results_df.head())
    
    # 保存CSV文件
    output_parent = os.path.dirname(output_folder)
    csv_path = os.path.join(output_parent, "响应统计结果.csv")
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"结果已保存: {csv_path}")
    
    # 打印统计信息
    folder_count = len(set(results_df['folder']))
    processed_metrics = list(set(results_df['metric']))
    print(f"\n处理完成!")
    print(f"共处理了 {folder_count} 个角度的数据")
    print(f"处理的指标: {', '.join(processed_metrics)}")
    
    if angles:
        print(f"角度范围: {min(angles)}° - {max(angles)}°")
    
    # 计算极值范围
    print(f"\n各指标极值范围:")
    for metric in processed_metrics:
        metric_data = results_df[results_df['metric'] == metric]
        if not metric_data.empty:
            min_extreme = metric_data['extreme'].min()
            max_extreme = metric_data['extreme'].max()
            print(f"  {metric}: {min_extreme:.2f} ~ {max_extreme:.2f}")
else:
    print("没有生成有效的结果数据")

