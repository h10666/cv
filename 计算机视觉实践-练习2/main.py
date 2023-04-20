import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize, ToTensor

writer = SummaryWriter()


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.maxpool1(torch.relu(self.conv1(x)))
        x = self.maxpool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainer(model, trainloader, testloader, writer, criterion, config):
    """
    训练函数
    :param model:模型
    :param trainloader: 训练集
    :param testloader: 测试集
    :param writer: tensorboard
    :param criterion: 损失函数
    :param config: 相关超参数
    :return: None
    """
    device = config['device']
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    last_loss = -10005
    best_accuracy = 0
    batches = len(trainloader.dataset) // config['batch_size']
    for epoch in range(config['epochs']):
        model.train()
        for batch, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if last_loss < -10000:
                last_loss = loss.item()
            else:
                last_loss = last_loss * 0.99 + loss.item()

            writer.add_scalar("Loss/Train", last_loss, batch + epoch * batches)

        valid_pred_true, valid_set_size = 0, 0
        test_loss_sum = 0.
        whole_pred = torch.tensor([]).to(device)
        whole_target = torch.tensor([]).to(device)
        for imgs, labels in testloader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                pred = model(imgs)
                loss = criterion(pred, labels)
                _, predicted = torch.max(pred.data, dim=1)
                whole_pred = torch.cat((whole_pred, predicted))
                whole_target = torch.cat((whole_target, labels))
            for i in range(len(labels)):
                if (int(predicted[i]) == int(labels[i])):
                    valid_pred_true += 1
                valid_set_size += 1
            test_loss_sum += loss.item()
        accuracy = valid_pred_true / valid_set_size
        if best_accuracy < accuracy:
            torch.save(model.state_dict(), config['save_path'])
        writer.add_scalar("Loss/Test", test_loss_sum, epoch)
        writer.add_scalar('Results/Precision', accuracy, epoch)
        print("epoch: {:}, Train Loss = {:.5f}, Test Loss = {:.5f}, Precision: {:.5f}".format(epoch, last_loss,
                                                                                              test_loss_sum, accuracy))


config = {
    'device': 'cuda:7',
    'batch_size': 64,
    'learning_rate': 1e-3,
    'save_path': 'mnist.pt',
    'epochs': 30
}


#数据增强
train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

#数据集加载
train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=train_transform, download=True)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], pin_memory=True)


#模型定义和损失函数定义
model = LeNet()
criterion = nn.CrossEntropyLoss(label_smoothing=0.3)

trainer(model=model, trainloader=train_loader, testloader=test_loader, writer=writer, criterion=criterion,
        config=config)

all_preds = []
all_targets = []
# model.load_state_dict(torch.load('mnist.pt'))
# model = model.to(config['device'])

##绘制混淆矩阵
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(config['device'])
        labels = labels.to(config['device'])
        preds = model(images)
        all_preds.extend(torch.argmax(preds, dim=1).cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
conf_mat = confusion_matrix(all_targets, all_preds)
print(conf_mat)

import matplotlib.pyplot as plt
import seaborn as sns

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.figure(figsize=(8, 8))
sns.heatmap(conf_mat, annot=True, cmap='Blues', xticklabels=classes, yticklabels=classes, fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.jpg')
