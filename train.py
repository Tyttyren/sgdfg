import os
import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
import numpy as np


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # transform2 = transforms.Compose(
    #     [transforms.Resize(size=(32, 32)),
    #      transforms.Normalize(mean=[127.5], std=[127.5])]
    # )
    #
    # train_set2 = l.MyDataset()

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data1', train=True,
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data1', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    train(train_loader)
    # val_loader = train_loader
    # val_set = train_set
    # val_data_iter = iter(val_loader)
    # val_image, val_label = next(val_data_iter)
    #
    # # classes = ('plane', 'car', 'bird', 'cat',
    # #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # net = LeNet()
    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=0.001)
    #
    # for epoch in range(5):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for step, data in enumerate(train_loader, start=0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = loss_function(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if step % 500 == 499:  # print every 500 mini-batches
    #             with torch.no_grad():
    #                 outputs = net(val_image)  # [batch, 10]
    #                 predict_y = torch.max(outputs, dim=1)[1]
    #                 accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
    #
    #                 print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
    #                       (epoch + 1, step + 1, running_loss / 500, accuracy))
    #                 running_loss = 0.0
    #
    # print('Finished Training')
    #
    # save_path = './模型1.pth'
    # torch.save(net.state_dict(), save_path)
import matplotlib.pyplot as plt
def draw(value, label, x_name, y_name, title):   
    print('draw',title) 
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(label, value, c='red', label="train")
    # plt.plot(label, value[1], c='green', linestyle='--', label="test")
    demo = value[1]
    plt.scatter(label, value, c='red')
    plt.legend(loc='best')
    plt.ylim([np.min(demo)/2,np.max(demo)*1.5])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel(x_name, fontdict={'size': 16})
    plt.ylabel(y_name, fontdict={'size': 16})
    plt.title(title, fontdict={'size': 20})
    plt.show()


def train(train_loader):
    val_loader = train_loader
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # t1 = summary(net,(1,32,32))
    # print(t1)
    acc = []
    loss1 = []
    label_list = []
    for epoch in range(10):  # loop over the dataset multiple times
        label_list.append(epoch)
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data


            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            # print(labels)
            # labels = torch.tensor(labels)

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 50 == 0:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 100, accuracy))
                    acc.append(accuracy)
                    loss1.append(running_loss/100)
                    running_loss = 0.0

    print('Finished Training')
    draw(value=acc,label=label_list,x_name='epochs',y_name="acc",title="Accuracy")
    draw(value=loss1,label=label_list,x_name='epochs',y_name="Loss",title="Loss")
    os.pause()
    save_path = '../Leng1.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()