import matplotlib.pyplot as plt
import numpy as np
log_file = 'Train.log'

epochs = []
loss_values = []
val_auc_values = []
test_loss_values = []
test_acc_values = []
test_pre_values = []
test_rec_values = []
test_auc_values = []

with open(log_file, 'r') as file:
    for line in file.readlines()[4:]:
        parts = line.split(',')
        epoch = int(parts[0].split('-')[5])
        loss = float(parts[1].split('-')[1])
        val_auc = float(parts[2].split('-')[1])
        test_loss = float(parts[3].split('-')[1])
        test_acc = float(parts[4].split('-')[1])
        test_pre = float(parts[5].split('-')[1])
        test_rec = float(parts[6].split('-')[1])
        test_auc = float(parts[7].split('-')[1])

        epochs.append(epoch)
        loss_values.append(loss)
        val_auc_values.append(val_auc)
        test_loss_values.append(test_loss)
        test_acc_values.append(test_acc)
        test_pre_values.append(test_pre)
        test_rec_values.append(test_rec)
        test_auc_values.append(test_auc)

# Plotting the data
plt.figure(figsize=(10, 6))
# plt.plot(epochs, loss_values, label='Train Loss')
# plt.plot(epochs, val_auc_values, label='Val Accuracy')
# plt.plot(epochs, test_loss_values, label='Test Loss')
# plt.plot(epochs, test_acc_values, label='Test Accuracy')
# plt.plot(epochs, test_pre_values, label='Test Precision')
# plt.plot(epochs, test_rec_values, label='Test Recall')
plt.plot(epochs, test_auc_values, label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Test Metrics over Epochs')
plt.legend()
plt.grid(True)

tick_values = np.arange(epochs[0]-1, epochs[-1]+1, 20)
tick_labels = [str(epoch) if epoch % 20 == 0 else '' for epoch in tick_values]
plt.xticks(tick_values, tick_labels)

plt.yticks(np.arange(0, 1, 0.1))



save_path = 'log_to_picture_TestAUC.png'
plt.savefig(save_path, dpi=300)

plt.show()

