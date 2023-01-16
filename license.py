import sys
sys.path.append('D:/soft/python_project/python_project/lenet')
import utils
import random
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
import train


class MyDataset(Dataset):
    ''' 自定义的数据集类 '''

    def __init__(self, label_list, transform=None, csv_path='./a.csv'):
        '''
        * `label_list`: 标签与文件路径的映射列表
        * `transform`：数据处理函数
        '''
        super(MyDataset, self).__init__()
        # random.shuffle(label_list)      # 打乱映射列表
        self.classes = utils.categories
        self.label_list = label_list
        self.transform = transform
        # self._csv_file = os.path.join(csv_path)
        # self._categories = utils.total_cate
        self.data_set = []
        for key in label_list.keys():
            cur_path = label_list[key]
            img_name = os.listdir(cur_path)
            for name in img_name:
                img_path = cur_path + '/' + name
                self.data_set.append((img_path, key))

    def __getitem__(self, index):
        img_path, label = self.data_set[index]
        img = Image.open(img_path)
        # Image.Image.show(img)
        # np_array = np.array(img)
        out = self.transform(img)
        if utils.start01(label):
            loca = utils.privance[label]
            real_label = utils.total_cate[loca][0]
        else:
            real_label = utils.total_cate[label][0]
        # print(out)
        return out, int(real_label)

    def __len__(self):
        ''' 获取数据集样本总数 '''
        return len(self.data_set)


# print(train_dataset.classes)
# test_dataset = MyDataset('D:\Pycharm\Project\lenet\test', transform2)    # 测试集

# list = os.listdir('D:\Pycharm\Project\lenet\train')
# train_dataset.__getitem__()
# print(list)
# print(train_dataset)
#

if __name__ == '__main__':
    train_dataset = MyDataset(utils.add_file_to_list('D:/soft/python_project/python_project/lenet/train'),
                              utils.transform)  # 训练集
    a = DataLoader(dataset=train_dataset, shuffle=True, batch_size=256)
    train.train(a)
