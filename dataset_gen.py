import os
import numpy as np
import pandas as pd
import torch
from PIL import Image  # PIL库的Image包是基于python开发的数字图片处理包。
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms  # 使用torchvision中的变换

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练和验证集采用transform做变换
transform = transforms.Compose([
    transforms.Resize([256, 256]),  # 图片resize
    # transforms.RandomCrop(224),  # 随机裁剪224*224，但此处不能随机裁剪，因为坐标群不能及时变化
    # transforms.RandomHorizontalFlip(),    # 水平翻转，但此处不能随机裁剪，因为坐标群不能及时变化
    transforms.ToTensor(),  # 将图像转为Tensor,数据归一化了欸！img.float().div(255)
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化,为了展示图像，先注释掉
])
# 测试集采用test_transform做变换
test_transform = transforms.Compose([
    transforms.Resize([288, 288]),  # 把图片resize为256*256
    transforms.RandomCrop(256),  # 随机裁剪224*224，测试时无标签
    transforms.RandomHorizontalFlip(),    # 水平翻转
    transforms.ToTensor(),  # 将图像转为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])   # 标准化
])


class KeyPointsDataSet(Dataset):
    """服装关键点标记数据集"""

    def __init__(self, root_dir, image_set='train', transforms=None):
        """
        初始化数据集
        :param root_dir: 数据目录(.csv和images的根目录)
        :param image_set: train训练,val验证,test测试
        :param transforms（callable,optional）:图像变换-可选
        标签数据文件格式为csv_file: 标签csv文件(内容：图像相对地址-category类型-标签coordination坐标)
        """
        self._imgset = image_set
        self._image_paths = []  # 用于存储图片地址列表
        self._labels = []  # 图片标签坐标群
        self._cates = []  # 标签：服装类别
        self._csv_file = os.path.join(root_dir, image_set + '.csv')  # csv标签文件地址
        self.__getFileList()  # 获取数据(图像，坐标，类型)
        self._categories = ['blouse', 'outwear', 'dress', 'trousers', 'skirt', ]
        self._root_dir = root_dir
        self._transform = transforms

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        img_id = self._image_paths[idx]
        img_id = os.path.join(self._root_dir, img_id)
        image = Image.open(img_id).convert('RGB')  # [3, 256, 256](通道数,高,宽）= (c, h, w)
        imgSize = image.size  # 原始图像宽高
        label = np.asfortranarray(self._labels[idx])  # (x, y, 显隐)=(宽，高，显隐性)
        category = self._categories.index(self._cates[idx])  # 0,1,2,3,4

        if self._transform:
            image = self._transform(image)  # 返回torch.Size([3, 256, 256])
        else:
            image.resize((256, 256))  # 使用resize
        afterSize = image.numpy().shape[1:]  # 缩放后图像的宽高
        # print(imgSize, afterSize)
        bi = np.array(afterSize) / np.array(imgSize)
        label[:, 0:2] = label[:, 0:2] * bi

        return image, label, category

    def __getFileList(self):
        file_info = pd.read_csv(self._csv_file)
        self._image_paths = file_info.iloc[:, 0]  # 第一列，相对地址列
        self._cates = file_info.iloc[:, 1]  # 第二列，服装类型：blouse,trousers,skirt,dress,outwear
        if self._imgset == 'train':
            landmarks = file_info.iloc[:, 2:26].values  # panda中DataFrame数据的读取。第3-25列为坐标群，共24组坐标，
            for i in range(len(landmarks)):  # 处理坐标数据84_497_1 to [84,497,1]
                label = []
                for j in range(24):
                    plot = landmarks[i][j].split('_')
                    coor = []
                    for per in plot:
                        coor.append(int(per))
                    label.append(coor)
                self._labels.append(np.concatenate(label))
            self._labels = np.array(self._labels).reshape((-1, 24, 3))
        else:
            self._labels = np.ones((len(self._image_paths), 24, 3)) * (-1)


def showImageAndCoor(img, coords):
    for coor in coords:
        if coor[2] == -1:
            pass
        else:
            img[:, coor[1]-1, coor[0]] = [255, 0, 0]  # (y,x)
            img[:, coor[1], coor[0]] = [255, 0, 0]  # 设置关键点位置坐标为红色，为了便于观察，将关键点四领域都设置为红色
            img[:, coor[1]+1, coor[0]] = [255, 0, 0]
            img[:, coor[1], coor[0]-1] = [255, 0, 0]
            img[:, coor[1], coor[0]+1] = [255, 0, 0]
    # 因为Image不能直接接受（3，256，256）多维数据，需要逐个击破
    # img = Image.fromarray(img * 255, mode='RGB')  # 所以这样写只能得到一条线，呵呵
    img0 = Image.fromarray(255 * img[0]).convert('L')
    img1 = Image.fromarray(255 * img[1]).convert('L')
    img2 = Image.fromarray(255 * img[2]).convert('L')
    img = Image.merge("RGB", [img0, img1, img2])

    img.show()


if __name__ == "__main__":
    fashionDataset = KeyPointsDataSet(root_dir=r"D:\Pycharm\Project\lenet\data",
                                      image_set="train",
                                      transforms=transform,
                                      )
    dataloader = DataLoader(dataset=fashionDataset, batch_size=4)
    for i_batch, data in enumerate(dataloader, 0):
        img, label, category = data
        img, label, category = img.numpy(), label.numpy(), category.numpy()  # 'torch.Tensor'不能直接显示，需要转换程io能处理的numpy数组格式。
        print(img.shape, label.shape, category)
        showImageAndCoor(img[0], label[0])
        break