import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import tensorflow as tf
from tf_cfc import CfcCell, MixedCfcCell
import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, accuracy_score,f1_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



BEST_MIXED = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 64,
    "embed_dim": 32,
    "embed_dr": 0.3,
    "epochs": 20,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00029,
    "use_mixed": True,
}
# 87.04% (MAX)
#  85.91% $\pm$ 0.99
BEST_DEFAULT = {
    "clipnorm": 10,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 192,
    "embed_dim": 192,
    "embed_dr": 0.0,
    "epochs": 47,
    "base_lr": 0.0005,
    "decay_lr": 0.7,
    "backbone_activation": "silu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 2,
    "weight_decay": 3.6e-05,
    "use_mixed": False,
    "no_gate": False,
}
# 87.52\% $\pm$ 0.09
BEST_NO_GATE = {
    "clipnorm": 5,
    "optimizer": "rmsprop",
    "batch_size": 128,
    "size": 224,
    "embed_dim": 192,
    "embed_dr": 0.2,
    "epochs": 50,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "silu",
    "backbone_dr": 0.1,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 2.7e-05,
    "use_mixed": False,
    "no_gate": True,
    "minimal": False,
}
# 81.72\% $\pm$ 0.50
BEST_MINIMAL = {
    "clipnorm": 1,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 320,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 27,
    "base_lr": 0.0005,
    "decay_lr": 0.8,
    "backbone_activation": "relu",
    "backbone_dr": 0.0,
    "backbone_units": 64,
    "backbone_layers": 1,
    "weight_decay": 0.00048,
    "use_mixed": False,
    "no_gate": False,
    "minimal": True,
}
# 61.76\% $\pm$ 6.14
BEST_LTC = {
    "clipnorm": 10,
    "optimizer": "adam",
    "batch_size": 128,
    "size": 128,
    "embed_dim": 64,
    "embed_dr": 0.0,
    "epochs": 50,
    "base_lr": 0.05,
    "decay_lr": 0.95,
    "backbone_activation": "lecun",
    "backbone_dr": 0.0,
    "forget_bias": 2.4,
    "backbone_units": 128,
    "backbone_layers": 1,
    "weight_decay": 1e-05,
    "use_mixed": False,
    "no_gate": False,
    "minimal": False,
    "use_ltc": True,
}


# dataset = "PAD4_new_004_random1"
# model_path = 'save/model/20241127204415/26_BEST_NO_GATE_epoch_27_Accuracy_0.9930210.h5'  # 请替换为您的模型路径

# dataset = "PAD4_new_004_random2"
# model_path = 'save/model/20241127204938/24_BEST_NO_GATE_epoch_25_Accuracy_0.9920239.h5'  # 请替换为您的模型路径

# dataset = "PAD4_new_004_random3"
# model_path = 'save/model/20241127210417/21_BEST_NO_GATE_epoch_22_Accuracy_0.9900299.h5'  # 请替换为您的模型路径


# dataset = "PAD4_new_004_random4"
# model_path = 'save/model/20241127205617/23_BEST_NO_GATE_epoch_24_Accuracy_0.9900398.h5'  # 请替换为您的模型路径

# dataset = "PAD4_new_004_random5"
# model_path = 'save/model/20241127210649/25_BEST_NO_GATE_epoch_26_Accuracy_0.9930210.h5'  # 请替换为您的模型路径

# dataset = "PAD4_new_004_random6"
# model_path = 'save/model/20241127211302/25_BEST_NO_GATE_epoch_26_Accuracy_0.9880359.h5'  # 请替换为您的模型路径

# dataset = "PAD4_new_004_random7"
# model_path = 'save/model/20241127211638/25_BEST_NO_GATE_epoch_26_Accuracy_0.9950150.h5'  # 请替换为您的模型路径


# dataset = "PAD4_new_004_random8"
# model_path = 'save/model/20241127211737/25_BEST_NO_GATE_epoch_26_Accuracy_0.9970090.h5'  # 请替换为您的模型路径

# dataset = "human_PAD_new_004"
# model_path = 'save/model/human_PAD_new_004_20250416135431/49_BEST_NO_GATE_epoch_50_Accuracy_0.9413076.h5'


dataset = "bindingDB"
model_path = "save/model/bindingDB/20250416163427/49_BEST_NO_GATE_epoch_50_Accuracy_0.9484982.h5"






model_config = BEST_NO_GATE #选择所加载模型的模型架构类别，可以从model_path中获取



maxlen = 192
# 定义模型架构构建函数
def build_model(config):
    if config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)
    else:
        cell = CfcCell(units=config["size"], hparams=config)

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    cell_input = tf.expand_dims(inputs, axis=1)

    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(10)
    output_states = rnn(cell_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs, y)
    return model





# 载入测试数据
test_data = pd.read_csv(f'../gcn_part/feature/{dataset}/test/protein_ligant_label_features.csv', header=None)
test_x = test_data.iloc[:, :-1].values

# 构建模型（使用与训练时相同的配置）
loaded_model = build_model(model_config)

# 载入已训练的模型参数

loaded_model.load_weights(model_path)

# 进行预测
predictions = loaded_model.predict(test_x)

# 根据需要对预测结果进行处理
# 例如，如果您的模型输出是概率分布，您可以选择最大概率的类别作为预测结果
predicted_labels = np.argmax(predictions, axis=1)

# 打印预测结果
print("Predicted Labels:", predicted_labels)

# 评估预测结果
predictions_list = predicted_labels.tolist()
test_label_list = test_data.iloc[0:,-1].values.tolist()

# predict_true_n = 0
# predict_false_n = 0
# for i in range(0,len(predictions_list)):
#     if predictions_list[i] == test_label_list[i]:
#         predict_true_n += 1
#     else:
#         predict_false_n += 1
#
# print(f"一共{len(predictions_list)}个样本，预测正确{predict_true_n}个，预测错误{predict_false_n}个")

# 计算Precision（精确度）
precision = precision_score(test_label_list, predictions_list, average='binary')  # 如果是二分类
print(f"Precision: {precision:.4f}")

# 计算Recall（召回率）
recall = recall_score(test_label_list, predictions_list, average='binary')  # 如果是二分类
print(f"Recall: {recall:.4f}")

# 计算混淆矩阵
tn, fp, fn, tp = confusion_matrix(test_label_list, predictions_list).ravel()

# 计算Sensitivity (SE) (召回率)
sensitivity = tp / (tp + fn)
print(f"Sensitivity (SE): {sensitivity:.4f}")

# 计算Specificity (SP) (特异性)
specificity = tn / (tn + fp)
print(f"Specificity (SP): {specificity:.4f}")

# 计算Accuracy (ACC) (准确率)
accuracy = accuracy_score(test_label_list, predictions_list)
print(f"Accuracy (ACC): {accuracy:.4f}")

f1 = f1_score(test_label_list, predictions_list, average='binary')  # 如果是多分类，需调整average参数
print(f"F1 Score: {f1:.4f}")


if len(np.unique(test_label_list)) == 2:  # 只有两个类别时才计算AUC
    auc = roc_auc_score(test_label_list, predictions_list)
    print(f"AUC: {auc:.4f}")
else:
    print("AUC calculation is not supported for multi-class classification.")








