from torch import nn
import torch
import torch.nn.functional as F

'''
    说明:
    1.LeNet是5层网络
    2.nn.ReLU(inplace=True)  参数为True是为了从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量
    3.本模型的维度注释均省略了N(batch_size)的大小,即input(3, 32, 32)-->input(N, 3, 32, 32)
    4.nn.init.xavier_uniform_(m.weight)
     用一个均匀分布生成值,填充输入的张量或变量,结果张量中的值采样自U(-a, a)，
     其中a = gain * sqrt( 2/(fan_in + fan_out))* sqrt(3),
     gain是可选的缩放因子,默认为1
     'fan_in'保留前向传播时权值方差的量级,'fan_out'保留反向传播时的量级
    5.nn.init.constant_(m.bias, 0)
      为所有维度tensor填充一个常量0
'''


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5, 1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5, 1)
        # self.pool2 = nn.MaxPool2d(1, 1)
        # self.conv3 = nn.Conv2d(16, 120, 5, 1)
        # self.fc1 = nn.Linear(120*6*6, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 66)
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 66)

    def forward(self, x):
        # x = F.relu(self.conv1(x))  # C1输出维度为28*28*6
        # x = self.pool1(x)          # S2输出维度为14*14*6
        # x = F.relu(self.conv2(x))  # C3输出维度为10*10*6
        # x = self.pool2(x)          # S4输出维度为5*5*16
        # x = F.relu(self.conv3(x))  # C5输出维度为120
        # x = x.view(-1, 120*6*6)
        # x = F.relu(self.fc1(x))    # F6输出维度为84
        # x = F.dropout(x, p=0.25)
        # y = self.fc2(x)
        # y = self.fc3(y)# F7输出维度为10
        # return y
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.maxpooling = torch.nn.MaxPool2d(2)
        self.avgpool = torch.nn.AvgPool2d(2)
        self.globalavgpool = torch.nn.AvgPool2d((8, 10))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.dropout50 = torch.nn.Dropout(0.5)
        self.dropout10 = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(256, 40)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.maxpooling(x)
        x = self.dropout10(x)
        x = self.bn2(F.relu(self.conv3(x)))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.maxpooling(x)
        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        return x