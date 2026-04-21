import torch
import numpy as np

# 替换为你的.pth文件路径
pth_path = "/EVAL/Detail/CAMO_ours.pth"

# 读取.pth文件（返回的是之前保存的Res字典）
res_dict = torch.load(pth_path)

# 1. 查看所有指标的键名
print(".pth文件包含的指标：", res_dict.keys())

# 2. 查看标量指标（比如MAE、MaxFm）
print("MAE值：", res_dict['MAE'])
print("MaxFm值：", res_dict['MaxFm'])
print("S-measure值：", res_dict['Sm'])

# 3. 查看数组指标（比如Prec/Recall的长度和前5个值）
print("Prec数组长度：", len(res_dict['Prec']))
print("Prec前5个值：", res_dict['Prec'][:5])
print("Recall前5个值：", res_dict['Recall'][:5])