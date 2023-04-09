from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Normalize, ToTensor, CenterCrop, Resize, RandomHorizontalFlip, \
    RandomVerticalFlip, ToPILImage, RandomRotation
from torch.utils.tensorboard import SummaryWriter
import os
import torch.optim as optim
import torch

writer = SummaryWriter()
train_transform = Compose([
    # Resize((256, 256)),
    # RandomHorizontalFlip(p=0.5),
    # RandomVerticalFlip(p=0.5),
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=4 * 4 * 16, out_features=120)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=120, out_features=84)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=84, out_features=10)
        )

    def forward(self, input):
        conv1_output = self.conv1(input)  # [28,28,1]-->[24,24,6]-->[12,12,6]
        conv2_output = self.conv2(conv1_output)  # [12,12,6]-->[8,8,16]-->[4,4,16]
        conv2_output = conv2_output.view(-1, 4 * 4 * 16)  # [n,4,4,16]-->[n,4*4*16],其中n代表个数
        fc1_output = self.fc1(conv2_output)  # [n,256]-->[n,120]
        fc2_output = self.fc2(fc1_output)  # [n,120]-->[n,84]
        fc3_output = self.fc3(fc2_output)  # [n,84]-->[n,10]
        return fc3_output


config = {
    'device': 'cuda:6',
    'batch_size': 128,
    'learning_rate': 1e-3,
    'save_path': 'mnist.pt',
    'epochs': 1000

}


def trainer(model, trainloader, testloader, writer, criterion, config):
    # if not os.path.isdir('./models'):
    #     os.mkdir('./models')
    device = config['device']
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    last_loss = -10005
    best_accuracy = 0
    for epoch in range(config['epochs']):
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

            writer.add_scalar("Loss/Train", last_loss, batch)

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
        writer.add_scalar("Loss/Test", last_loss, batch)
        writer.add_scalar('Results/Precision', accuracy, epoch)
        print("epoch: {:}, Loss = {:.5f}, Precision: {:.5f}".format(epoch, loss.item(), accuracy))


train_data = torchvision.datasets.MNIST(root="./data", train=True, transform=train_transform, download=True)
test_data = torchvision.datasets.MNIST(root="./data", train=False, transform=train_transform, download=True)

train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], pin_memory=True)

# img = Image.fromarray((train_data[0][0].numpy() * 255).astype('uint8').squeeze(), mode='L')
model = LeNet()
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

trainer(model=model, trainloader=train_loader, testloader=test_loader, writer=writer, criterion=criterion,
        config=config)
