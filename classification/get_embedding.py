import torch
import os
from model import GCN
from dataset import *
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import numpy as np
import csv


# 加载模型参数
model = GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2)
model.load_state_dict(torch.load('epoch-60.pt'))
model.eval()
# 准备输入数据
data_root = 'data'
DATASET = 'P40989'
fpath = os.path.join(data_root, DATASET)
# 加载数据
train_set = GNNDataset(fpath, types='train')
val_set = GNNDataset(fpath, types='val')
test_set = GNNDataset(fpath, types='test')

# 处理并写入val/csv文件中(蛋白质已完成)
csv_file = 'feature/P40989/val/protein_features.csv'
folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in val_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            protein_x_list = protein_x.flatten().tolist()
        writer.writerow(protein_x_list)

# 处理并写入val/csv文件中(配体已完成)
csv_file = 'feature/P40989/val/ligant_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in val_set:
        with torch.no_grad():
            ligand_x = model.ligand_encoder(data)
            ligand_x_list = ligand_x.flatten().tolist()
        writer.writerow(ligand_x_list)


#处理并写入val/csv文件中(复合已完成)
csv_file = 'feature/P40989/val/protein_ligant_label_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in val_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            ligand_x = model.ligand_encoder(data)

            protein_x_list = protein_x.flatten().tolist()
            ligand_x_list = ligand_x.flatten().tolist()
            label_lsit = data.y.flatten().tolist()

            complex_list = protein_x_list + ligand_x_list + label_lsit
        writer.writerow(complex_list)

# 处理并写入train/csv文件中(蛋白质已完成)
csv_file = 'feature/P40989/train/protein_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in train_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            protein_x_list = protein_x.flatten().tolist()
        writer.writerow(protein_x_list)

# 处理并写入train/csv文件中(配体已完成)
csv_file = 'feature/P40989/train/ligant_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in train_set:
        with torch.no_grad():
            ligand_x = model.ligand_encoder(data)
            ligand_x_list = ligand_x.flatten().tolist()
        writer.writerow(ligand_x_list)


#处理并写入train/csv文件中(复合已完成)
csv_file = 'feature/P40989/train/protein_ligant_label_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in train_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            ligand_x = model.ligand_encoder(data)

            protein_x_list = protein_x.flatten().tolist()
            ligand_x_list = ligand_x.flatten().tolist()
            label_lsit = data.y.flatten().tolist()

            complex_list = protein_x_list + ligand_x_list + label_lsit
        writer.writerow(complex_list)

# 处理并写入test/csv文件中(蛋白质已完成)
csv_file = 'feature/P40989/test/protein_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in test_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            protein_x_list = protein_x.flatten().tolist()
        writer.writerow(protein_x_list)

# 处理并写入test/csv文件中(配体已完成)
csv_file = 'feature/P40989/test/ligant_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in test_set:
        with torch.no_grad():
            ligand_x = model.ligand_encoder(data)
            ligand_x_list = ligand_x.flatten().tolist()
        writer.writerow(ligand_x_list)


#处理并写入test/csv文件中(复合已完成)
csv_file = 'feature/P40989/test/protein_ligant_label_features.csv'

folder_path = os.path.dirname(csv_file)
# 创建目标文件夹（如果不存在）
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for data in test_set:
        with torch.no_grad():
            protein_x = model.protein_encoder(data.target)
            ligand_x = model.ligand_encoder(data)

            protein_x_list = protein_x.flatten().tolist()
            ligand_x_list = ligand_x.flatten().tolist()
            label_lsit = data.y.flatten().tolist()

            complex_list = protein_x_list + ligand_x_list + label_lsit
        writer.writerow(complex_list)


print("运行完毕")