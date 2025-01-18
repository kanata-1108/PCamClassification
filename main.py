import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from model import My_model
import matplotlib.pyplot as plt

# 独自のDataset
class MakeDataset(Dataset):
    def __init__(self, x_path, y_path, transform):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
        self.transform = transform
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        x = torch.permute(x, (2, 0, 1))
        y = torch.FloatTensor(self.y[index])

        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.x)

def evaluation(net_model, loader):

    losses = 0
    accuracy = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for (inputs, labels) in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_model(inputs)
            loss = criterion(outputs, labels)
            losses += loss.item()
            predicted_label = torch.where(outputs < 0.5, 0, 1)
            accuracy += (predicted_label == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

        losses /= len(loader)
        accuracy /= len(loader.dataset)
        auc = roc_auc_score(all_labels, all_preds)

    return losses, accuracy, auc

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.chdir(os.path.dirname(__file__))

    x_train_path = "./data/camelyonpatch_level_2_split_train_x.npy"
    y_train_path = "./data/camelyonpatch_level_2_split_train_y.npy"
    x_valid_path = "./data/camelyonpatch_level_2_split_valid_x.npy"
    y_valid_path = "./data/camelyonpatch_level_2_split_valid_y.npy"

    # 使用するデータの作成
    transform_train = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomInvert(p = 0.5),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    transform_valid = transforms.Compose([
            transforms.CenterCrop(64),
            transforms.Resize(224),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    train_dataset = MakeDataset(x_path = x_train_path, y_path = y_train_path, transform = transform_train)
    valid_dataset = MakeDataset(x_path = x_valid_path, y_path = y_valid_path, transform = transform_valid)

    train_dataset.x = train_dataset.x[:25000]
    train_dataset.y = train_dataset.y[:25000]

    BATCH_SIZE = 16
    train_loader = DataLoader(dataset = train_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = True)
    valid_loader = DataLoader(dataset = valid_dataset, batch_size = BATCH_SIZE, num_workers = 2, shuffle = False)

    epochs = 200
    model = My_model()
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 0.01, total_steps = epochs, pct_start = 0.1, anneal_strategy = 'cos')

    loss_value = np.zeros((epochs, 2))
    acc_value = np.zeros((epochs, 2))
    auc_value = np.zeros((epochs, 2))

    tmp_auc = 0

    print("-----train start-----")
    for epoch in range(epochs):
        
        # 学習
        model.train()
        for (inputs, labels) in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # 推論
        model.eval()
        train_loss, train_acc, train_auc = evaluation(model, train_loader)
        valid_loss, valid_acc, valid_auc = evaluation(model, valid_loader)

        # グラフ描画用配列にlossとaccを格納
        loss_value[epoch][0], loss_value[epoch][1] = train_loss, valid_loss
        acc_value[epoch][0], acc_value[epoch][1] = train_acc, valid_acc
        auc_value[epoch][0], auc_value[epoch][1] = train_auc, valid_auc

        # エポック毎のlossとacc
        print(f"[{epoch + 1}/{epochs}] :: train loss: {train_loss:.5f}, valid loss: {valid_loss:.5f}, train acc: {train_acc:.5f}, valid acc: {valid_acc:.5f}, train auc: {train_auc:.5f}, valid auc: {valid_auc:.5f}")

        # ValidデータのAUCがいいときにパラメータを保存
        if tmp_auc < valid_auc:
            tmp_auc = valid_auc

            torch.save(model.state_dict(), 'model_param_0118.pth')

    # グラフの描画&保存
    plt.plot(range(epochs), loss_value[:, 0], c = 'orange', label = 'train loss')
    plt.plot(range(epochs), loss_value[:, 1], c = 'blue', label = 'valid loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend()
    plt.title('loss(train n = 10000)')
    plt.savefig('./result/loss.png')
    plt.clf()

    plt.plot(range(epochs), acc_value[:, 0], c = 'orange', label = 'train acc')
    plt.plot(range(epochs), acc_value[:, 1], c = 'blue', label = 'valid acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend()
    plt.title('accuracy(train n = 10000)')
    plt.savefig('./result/acc.png')
    plt.clf()

    plt.plot(range(epochs), auc_value[:, 0], c = 'orange', label = 'train auc')
    plt.plot(range(epochs), auc_value[:, 1], c = 'blue', label = 'valid auc')
    plt.xlabel('epoch')
    plt.ylabel('AUC')
    plt.grid()
    plt.legend()
    plt.title('AUC(train n = 10000)')
    plt.savefig('./result/auc.png')