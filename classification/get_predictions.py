import torch
import os
from model import GCN
from dataset import *
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
model = GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)

# 加载模型参数
model.load_state_dict(torch.load('FinalModel/model_1.pt', map_location=device))
model.eval()

# 准备输入数据
# data_root = 'FinalModel/data'
data_root = 'data'
DATASET = 'PAD4_030'
fpath = os.path.join(data_root, DATASET)

# 加载数据
val_set = GNNDataset(fpath, types='val')

# 创建数据加载器
val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

# 进行预测
pred_list = []
pred_cls_list = []
label_list = []

for data in val_loader:
    with torch.no_grad():
        data = data.to(device)  # 将数据移到GPU
        pred = model(data)
        pred_cls = torch.argmax(pred, dim=-1)
        pred_prob = F.softmax(pred, dim=-1)
        pred_prob, indices = torch.max(pred_prob, dim=-1)
        pred_prob[indices == 0] = 1. - pred_prob[indices == 0]

        pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
        pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
        label_list.append(data.y.detach().cpu().numpy())

pred = np.concatenate(pred_list, axis=0)
pred_cls = np.concatenate(pred_cls_list, axis=0)
label = np.concatenate(label_list, axis=0)

# 合并数据为一个DataFrame
data_dict = {
    'pred': pred,
    'pred_cls': pred_cls,
    'label': label
}

df = pd.DataFrame(data_dict)

# 保存DataFrame为CSV文件
df.to_csv('save_predictions/PAD4_030/8.csv', index=False)

print("运行结束")




# # 进行预测
# pred_list = []
# pred_cls_list = []
# label_list = []
# for data in val_loader:
#     with torch.no_grad():
#         pred = model(data)
#         pred_cls = torch.argmax(pred, dim=-1)
#         pred_prob = F.softmax(pred, dim=-1)
#         pred_prob, indices = torch.max(pred_prob, dim=-1)
#         pred_prob[indices == 0] = 1. - pred_prob[indices == 0]
#
#         pred_list.append(pred_prob.view(-1).detach().cpu().numpy())
#         pred_cls_list.append(pred_cls.view(-1).detach().cpu().numpy())
#         label_list.append(data.y.detach().cpu().numpy())
#
# pred = np.concatenate(pred_list, axis=0)
# pred_cls = np.concatenate(pred_cls_list, axis=0)
# label = np.concatenate(label_list, axis=0)
#
# """
# pred是预测出每个DTI的概率
# pred_cls是将预测的概率二值化，即预测的结果列表
# label是数据的真实label
#
# pred_cls是预测值，label是真实值
# """
# # 合并数据为一个DataFrame
# data_dict = {
#     'pred': pred,
#     'pred_cls': pred_cls,
#     'label': label
# }
#
# df = pd.DataFrame(data_dict)
#
# # 保存DataFrame为CSV文件
# df.to_csv('save_predictions/30predictions.csv', index=False)
#
# print("运行结束")
