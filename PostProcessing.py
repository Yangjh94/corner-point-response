"""
本程序主要用来处理风振分析文件，提取各方向风振响应极值，并将结果输出到指定文件中。
"""
# 导入所需的库
import os
import pandas as pd
import matplotlib.pyplot as plt

# 指定 output文件路径
output_folder = os.path.join(os.getcwd(), "data", "output", "Timehistory_modal", "1-1", "Acceleration")

# 获取文件夹中的所有文件名称
file_names = os.listdir(output_folder)

# 初始化结果储存
results = []
# 遍历选中的文件夹
for file_name in file_names:
    # 获取子文件夹中所有的csv文件的名称
    suboutput_folder = os.path.join(output_folder, file_name)
    files_csv_names = [f for f in os.listdir(suboutput_folder) if f.endswith(".csv")]

    # 选择前5个csv文件进行处理
    for csv_file in files_csv_names[-5:]:
        # 拼接完整路径
        file_path = os.path.join(suboutput_folder, csv_file)
        print(f"正在处理文件: {file_path}")
        # 读取csv文件
        df = pd.read_csv(file_path)
        
        # 处理数据：计算每列的平均值
        mean_values = df.mean()
        std_values = df.std()

        # 将结果存储到列表中
        results.append({
            "file": csv_file,
            "folder": file_name,
            "mean": mean_values.to_dict(),
            "std": std_values.to_dict()
        })

results_df = pd.DataFrame(results)

# 将均值和标准差格式化为两位小数
results_df['mean'] = results_df['mean'].apply(lambda x: {k: round(v, 2) for k, v in x.items()})
results_df['std'] = results_df['std'].apply(lambda x: {k: round(v, 2) for k, v in x.items()})

# 将均值和标准差展开为单独的列
mean_df = pd.DataFrame(results_df['mean'].tolist()).add_prefix('mean_')
std_df = pd.DataFrame(results_df['std'].tolist()).add_prefix('std_')

# 合并均值和标准差列到结果表格
results_df = pd.concat([results_df.drop(['mean', 'std'], axis=1), mean_df, std_df], axis=1)
# 检查列名是否匹配
if not mean_df.columns.equals(std_df.columns):
    print("列名不匹配，调整列名。")
    std_df.columns = mean_df.columns

# 逐列计算均值和3.5倍标准差的相加结果
new_column = pd.DataFrame()
for col in mean_df.columns:
    new_column[f'new_{col}'] = mean_df[col] + 3.5 * std_df[col]

# 检查新列数据
print("New Column Data:")
print(new_column)

# 将新列合并到结果表格
results_df = pd.concat([results_df, new_column], axis=1)
print(results_df)

# 保存结果表格到文件到上一级目录中
output_folder = os.path.dirname(output_folder)
output_file_path = os.path.join(output_folder, "结果统计.csv")
results_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
print(f"结果表格已保存到: {output_file_path}")

