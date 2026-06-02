import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 不显示等级2以下的提示信息
from tf_cfc import CfcCell, MixedCfcCell
import tensorflow as tf
import datetime


dataset = "human_PAD_new_002"



"""
对gcn_part/feature/human_PAD_new_002进行训练.
"""



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



dataset = "human_PAD_new_002"


maxlen = 192
train_data = pd.read_csv(f'../gcn_part/feature/{dataset}/train/protein_ligant_label_features.csv', header=None)
train_x = train_data.iloc[0:, :-1].values
train_y = train_data.iloc[0:, -1].values

test_data = pd.read_csv(f'../gcn_part/feature/{dataset}/test/protein_ligant_label_features.csv', header=None)
test_x = test_data.iloc[0:, :-1].values
test_y = test_data.iloc[0:,-1].values






# (imdb_x_train, imdb_y_train), (imdb_x_val, imdb_y_val) = tf.keras.datasets.imdb.load_data(num_words=20000)
# imdb_x_train = tf.keras.preprocessing.sequence.pad_sequences(imdb_x_train, maxlen=200)
# imdb_x_val = tf.keras.preprocessing.sequence.pad_sequences(imdb_x_val, maxlen=200)
# print(imdb_x_train.shape[0])
# print(imdb_x_val.shape)

def eval(config, index_arg, verbose=0):
    if config["use_mixed"]:
        cell = MixedCfcCell(units=config["size"], hparams=config)

    else:
        cell = CfcCell(units=config["size"], hparams=config)

    inputs = tf.keras.layers.Input(shape=(maxlen,))
    cell_input = tf.expand_dims(inputs, axis=1)
    # RNN 层（如 LSTM、GRU）通常需要形状为 (batch_size, timesteps, features) 的输入。
    # 我们输入数据是 (batch_size, maxlen)，通过 tf.expand_dims 将其扩展为 (batch_size, 1, maxlen)，以适应 RNN 的输入要求。

    # cell_input = tf.keras.layers.Dropout(config["embed_dr"])(cell_input)
    rnn = tf.keras.layers.RNN(cell, time_major=False, return_sequences=False)
    dense_layer = tf.keras.layers.Dense(10)
    output_states = rnn(cell_input)
    y = dense_layer(output_states)

    model = tf.keras.Model(inputs, y)

    base_lr = config["base_lr"]
    decay_lr = config["decay_lr"]
    train_steps = train_x.shape[0] // config["batch_size"]

    learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
        base_lr, train_steps, decay_lr
    )
    opt = (
        tf.keras.optimizers.Adam
        if config["optimizer"] == "adam"
        else tf.keras.optimizers.RMSprop
    )

    optimizer = opt(learning_rate_fn, clipnorm=config["clipnorm"])
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    with tf.device('/GPU:0'):
        # Fit and evaluate
        hist = model.fit(
            x=train_x,
            y=train_y,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            validation_data=(test_x, test_y) if verbose else None,
            verbose=verbose,
            # callbacks=[CustomCallback()]
        )



        # test_accuracies = hist.history["val_sparse_categorical_accuracy"]
        # return np.max(test_accuracies)
        _, test_accuracy = model.evaluate(test_x, test_y, verbose=0)
    return test_accuracy


# 五折交叉验证

def score(config):
    acc = []
    for i in range(5):
        acc.append(100 * eval(config, i))
        # print(
        #     f" test accuracy [{len(acc)}/5]: {np.mean(acc):0.2f}\\% $\\pm$ {np.std(acc):0.2f}"
        # )
    print(f"result of 5fold : test accuracy: {np.mean(acc):0.2f}\\% $\\pm$ {np.std(acc):0.2f}")

# # 单次验证
# def score(config):
#     acc = []
#     acc.append(100 * eval(config, 0))
#
#
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def on_epoch_end(self, epoch, logs=None):
        _, test_accuracy = self.model.evaluate(test_x, test_y, verbose=0)
        print("Epoch {}: train Loss = {:.4f}, train Accuracy = {:.4f}, Test Accuracy = {:.4f}".format(epoch+1, logs["loss"], logs["sparse_categorical_accuracy"],test_accuracy))

        # 创建保存模型的文件夹
        save_folder = "save/model/{}".format(self.timestamp)
        os.makedirs(save_folder, exist_ok=True)
        # 保存模型的结构和配置
        model_name = "epoch_{}_Accuracy_{:.7f}".format(epoch+1, test_accuracy)
        model_path = os.path.join(save_folder, model_name)
        self.model.save(model_path,save_format='tf')


if __name__ == "__main__":
    score(BEST_NO_GATE)
    # score(BEST_LTC)
    # score(BEST_MINIMAL)
    # score(BEST_DEFAULT)
    # score(BEST_MIXED)