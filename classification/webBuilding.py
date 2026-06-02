import gradio as gr
import os
import shutil
import sys
from datetime import datetime

import os.path as osp
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from tqdm import tqdm
from model import GCN
from dataset import *
from torch_geometric.data import DataLoader
import torch.nn.functional as F


#FnalModel文件夹中哪个子文件夹的模型
FinalModel_1 = "2"

def upload_and_process(file):
    """
    根据用户上传的csv文件名来创建项目名和路径,并改名为xxx.csv，若上传xxx.csv,则原始文件保存为web_upload/xxx_20240829160306/data.csv
    """
    #获取文件名/不带扩展的文件名/时间
    formatted_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = os.path.basename(file.name)  # file_name=xxx.csv
    file_name_without_csv = os.path.splitext(file_name)[0]  # file_name_without_csv=xxx
    xxx_formatted_time = file_name_without_csv+'_'+formatted_time#xxx_20240829160306文件夹

    # 创建web_upload/xxx_20240829160306文件夹
    file_project=os.path.join('web_upload',xxx_formatted_time) #web_upload/xxx_20240829160306文件夹
    os.makedirs(file_project, exist_ok=True)

    #将用户上传的csv文件保存至web_upload/xxx_20240829160306文件夹并改名为data.csv
    file_path = os.path.join(file_project, 'data.csv')#web_upload/xxx_20240829160306/data.csv
    shutil.move(file.name, file_path)

    # 进行一系列处理
    # 这里假设处理函数是 process_file，且它将生成一个输出文件
    output_file_path = process_file(file_path,xxx_formatted_time)

    # 返回处理后的文件路径
    return output_file_path


# 示例处理函数
def process_file(input_file_path,xxx_formatted_time):
    source_file = input_file_path #web_upload/xxx_20240829160306/data.csv
    data_dir_2 = xxx_formatted_time     #xxx_20240829160306文件夹

    data_dir = 'data'
    data_dir_3 = os.path.join(data_dir, data_dir_2)#data/xxx_20240829160306文件夹
    save_dir_1 = os.path.join('save_predictions', data_dir_2)#save_predictions/xxx_20240829160306文件夹

    # 定义raw文件夹路径
    raw_dir = os.path.join(data_dir_3, 'raw')#data/xxx_20240829160306文件夹/raw
    destination_file = os.path.join(raw_dir, 'data.csv')#data/xxx_20240829160306文件夹/raw/data.csv

    # 检查并创建data/xxx_20240829160306文件夹
    if not os.path.exists(data_dir_3):
        os.makedirs(data_dir_3)
        print(f"文件夹 {data_dir_3} 已创建。")
    else:
        print(f"文件夹 {data_dir_3} 已经存在。")
        sys.exit()

    # 检查并创建data/xxx_20240829160306文件夹/raw文件夹
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        print(f"文件夹 {raw_dir} 已创建。")
    else:
        print(f"文件夹 {raw_dir} 已经存在。")
        sys.exit()

    # 复制#web_upload/xxx_20240829160306/data.csv到data/xxx_20240829160306文件夹/raw/data.csv
    if os.path.exists(source_file):
        shutil.copy(source_file, destination_file)
        print(f"文件 {source_file} 已成功复制到 {destination_file}。")
    else:
        print(f"源文件 {source_file} 不存在，无法复制。")

    # preprocessing_no_split.py部分
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------


    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def data_split_train_val_test(data_root='data', data_set='human'):
        data_path = osp.join(data_root, data_set, 'raw', 'data.csv')
        data_df = pd.read_csv(data_path)

        # Split data in train:val:test = 8:1:1 with the same random seed as previous study.
        # Please see https://github.com/masashitsubaki/CPI_prediction
        data_shuffle = data_df.sample(frac=1., random_state=1234)
        train_split_idx = int(len(data_shuffle) * 0.8)
        df_train = data_shuffle[:train_split_idx]
        df_val_test = data_shuffle[train_split_idx:]
        val_split_idx = int(len(df_val_test) * 0.5)
        df_val = df_val_test[:val_split_idx]
        df_test = df_val_test[val_split_idx:]

        df_train.to_csv(osp.join(data_root, data_set, 'raw', 'data_train.csv'), index=False)
        df_val.to_csv(osp.join(data_root, data_set, 'raw', 'data_val.csv'), index=False)
        df_test.to_csv(osp.join(data_root, data_set, 'raw', 'data_test.csv'), index=False)

        print(f"{data_set} split done!")
        print("Number of data: ", len(data_df))
        print("Number of train: ", len(df_train))
        print("Number of val: ", len(df_val))
        print("Number of test: ", len(df_test))

    '''
    Molecular graphs generation
    '''
    VOCAB_PROTEIN = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                     "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                     "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                     "U": 19, "T": 20, "W": 21,
                     "V": 22, "Y": 23, "X": 24,
                     "Z": 25}

    def seqs2int(target):
        return [VOCAB_PROTEIN[s] for s in target]

    def atom_features(atom):
        encoding = one_of_k_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti',
                                          'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt',
                                          'Hg',
                                          'Pb', 'Unknown'])
        encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2, 'other'])
        encoding += [atom.GetIsAromatic()]

        try:
            encoding += one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

        return np.array(encoding)

    def mol_to_graph(mol):
        features = []
        for atom in mol.GetAtoms():
            feature = atom_features(atom)
            features.append(feature / np.sum(feature))

        edges = []
        for bond in mol.GetBonds():
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

        if len(edges) == 0:
            return features, [[0, 0]]

        g = nx.Graph(edges).to_directed()
        edge_index = []
        for e1, e2 in g.edges:
            edge_index.append([e1, e2])

        return features, edge_index

    def one_of_k_encoding(x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))

    def one_of_k_encoding_unk(x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))

    class GNNDataset(InMemoryDataset):
        def __init__(self, root, types='train', transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)

            # 只加载data
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def raw_file_names(self):
            return ['data.csv']

        @property
        def processed_file_names(self):
            return ['processed_data.pt']

        def download(self):
            # Download to `self.raw_dir`.
            pass

        def _download(self):
            pass

        def process_data(self, data_path, graph_dict):
            df = pd.read_csv(data_path)

            data_list = []
            delete_list = []
            for i, row in df.iterrows():
                smi = row['compound_iso_smiles']
                sequence = row['target_sequence']
                label = row['affinity']

                if graph_dict.get(smi) == None:
                    print("Unable to process: ", smi)
                    delete_list.append(i)
                    continue

                x, edge_index = graph_dict[smi]

                target = seqs2int(sequence)
                target_len = 1200
                if len(target) < target_len:
                    target = np.pad(target, (0, target_len - len(target)))
                else:
                    target = target[:target_len]

                data = DATA.Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                    y=torch.FloatTensor([label]),
                    target=torch.LongTensor([target])
                )

                data_list.append(data)

            if len(delete_list) > 0:
                df = df.drop(delete_list, axis=0, inplace=False)
                df.to_csv(data_path, index=False)

            return data_list

        def process(self):
            df = pd.read_csv(self.raw_paths[0])
            smiles = df['compound_iso_smiles'].unique()

            graph_dict = dict()
            for smile in tqdm(smiles, total=len(smiles)):
                mol = Chem.MolFromSmiles(smile)
                if mol == None:
                    print("Unable to process: ", smile)
                    continue
                graph_dict[smile] = mol_to_graph(mol)

            train_list = self.process_data(self.raw_paths[0], graph_dict)

            if self.pre_filter is not None:
                train_list = [train for train in train_list if self.pre_filter(train)]

            if self.pre_transform is not None:
                train_list = [self.pre_transform(train) for train in train_list]

            print('Graph construction done. Saving to file.')

            # save preprocessed train data:
            data, slices = self.collate(train_list)
            torch.save((data, slices), self.processed_paths[0])

    GNNDataset(root=data_dir_3)

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # 修改processed名字部分
    processed_dir = os.path.join(data_dir_3, 'processed')

    # 检查processed文件夹是否存在
    if not os.path.exists(processed_dir):
        print(f"文件夹 {processed_dir} 不存在，程序终止。")
        sys.exit()

    source_file_processed_data = os.path.join(processed_dir, 'processed_data.pt')
    destination_file_processed_data_val = os.path.join(processed_dir, 'processed_data_val.pt')

    if os.path.exists(source_file_processed_data):
        os.rename(source_file_processed_data, destination_file_processed_data_val)
        print(f"文件 {source_file_processed_data} 已成功重命名为 {destination_file_processed_data_val}。")
    else:
        print(f"源文件 {source_file_processed_data} 不存在，无法重命名。")

    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    # 运行get_predictions.py的部分


    # 检查GPU是否可用，如果可用，则将模型和数据移到GPU上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 准备输入数据
    fpath = data_dir_3

    # 加载数据
    val_set = GNNDataset(fpath, types='val')

    # 创建数据加载器
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False)

    # 循环加载并运行模型1到模型8
    for i in range(1, 9):
        # 创建模型实例
        model = GCN(3, 25 + 1, embedding_size=128, filter_num=32, out_dim=2).to(device)

        # 加载模型参数
        model_path = f'FinalModel/{FinalModel_1}/model_{i}.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

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
        # 检查在save_predictions文件夹中创建save_predictions/xxx_20240829160306文件夹

        if not os.path.exists(save_dir_1):
            os.makedirs(save_dir_1)
            print(f"文件夹 {save_dir_1} 已创建。")

        save_dir_2 = os.path.join(save_dir_1, f"{i}.csv")
        df.to_csv(save_dir_2, index=False)

        print(f"model{i}运行结束，结果已保存为 {i}.csv")

    print("所有模型的运行结束")
    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------

    # 数据处理部分
    # 定义目录路径
    data_csv_path = source_file

    # 读取 data.csv 文件
    data_df = pd.read_csv(data_csv_path)
    # 遍历1到8的文件夹

    for i in range(1, 9):
        # 定义当前处理的csv文件路径
        csv_path = os.path.join(save_dir_1, f'{i}.csv')

        # 读取当前csv文件
        csv_df = pd.read_csv(csv_path)

        # 合并csv文件的label列与data.csv的index列
        merged_df = pd.merge(csv_df, data_df, left_on='label', right_on='index', how='left')

        # 删除不需要的列
        merged_df = merged_df.drop(columns=['affinity', 'label', 'pred_cls'])

        # 调整列的顺序
        final_df = merged_df[['index', 'compound_iso_smiles', 'target_sequence', 'pred']]

        # 将结果保存回原文件或另存为新文件
        merged_csv_path = os.path.join(save_dir_1, f'merged_{i}.csv')
        merged_df.to_csv(merged_csv_path, index=False)

        print(f"{i}.csv 已处理并保存为 merged_{i}.csv")
        # 初始化一个空的DataFrame，用于存储拼接后的数据

    combined_df = pd.DataFrame()
    # 遍历处理merged_1.csv到merged_8.csv
    for i in range(1, 9):
        # 定义当前处理的csv文件路径
        csv_path = os.path.join(save_dir_1, f'merged_{i}.csv')

        # 读取当前csv文件
        df = pd.read_csv(csv_path)

        # 新建model列，并设置其值为"model{i}"
        df['model'] = f'model{i}'

        # 将当前数据追加到combined_df中
        combined_df = pd.concat([combined_df, df], ignore_index=True)

        # 以index列和model列进行排序
        combined_df = combined_df.sort_values(by=['index', 'model'])

        # 将长表转置为宽表
        pivot_df = combined_df.pivot(index=['index', 'compound_iso_smiles', 'target_sequence'],
                                     columns='model',
                                     values='pred').reset_index()

    pivot_df['complex_score'] = (
            pivot_df['model1'] +
            pivot_df['model2'] +
            pivot_df['model3'] +
            pivot_df['model4'] +
            pivot_df['model5'] * 1.5 +
            pivot_df['model6'] * 1.5 +
            pivot_df['model7'] +
            pivot_df['model8'] * 1.5
    ).round(6)

    # 保存
    combined_csv_path = os.path.join(save_dir_1, 'result.csv')
    pivot_df.to_csv(combined_csv_path, index=False)


    print("所有数据已拼接完成，并保存在save_predictions/xxx_20240829160306文件夹/result.csv")



    # --------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------
    output_file_path = combined_csv_path
    return output_file_path





# 创建 Gradio 界面


iface = gr.Interface(
    fn=upload_and_process,
    inputs=gr.File(file_types=['csv'], label="请提交包含药物smiles和蛋白质序列的csv文件"),
    outputs=gr.File(label="下载结果文件"),
    title="Link_Li Lab Screening for PAD4"
)


# 启动 Gradio 接口
iface.launch(share=True,auth=[("pal3zz", "LHWlhw359"),
                              ("633","1234")]
             )

# iface.launch(auth=[("pal3zz", "LHWlhw359"),
#                               ("633","1234")]
#              )