import torch
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import DataLoader
import train
import cv2

transform1 = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
     # transforms.Grayscale(3),
     transforms.PILToTensor(),
     transforms.ConvertImageDtype(torch.float),
     # transforms.Normalize(mean=[127.5], std=[127.5]),
     transforms.ToPILImage()]
)


# i = Image.open('./data/SJUG5FN.jpg')
# # print(i.mode)
#
# img = Image.open('./train/C/4-5.jpg')
#
# res = transform1(img)
# Image.Image.show(res)
# print(res.mode)
# train1 = datasets.ImageFolder('./train', utils.transforms)

# data_loader = DataLoader(train1, batch_size=64, shuffle=True)
# train.train(data_loader)
# dict_a = ('1','abc')
#
# a = [dict_a]
# b,c = a[0]
# print(b,c)
# dataset = MyDataset(utils.add_file_to_list('D:/soft/python/python_project/lenet/train'), utils.transform)


# print(hasattr(dataset, "__getitems__"))


# dict1 = {1:"213",2:'1231'}
# print(dict1.__contains__(1))
# Blur the image for better edge detection
# img_blur = cv2.GaussianBlur(img,(3,3), SigmaX=0, SigmaY=0)
def img_show(img):
    Image.Image.show(Image.fromarray(img))

img = cv2.imread('D:\soft\python\python_project\lenet\img_file\img_3.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.GaussianBlur(img, (3,3), 0, 0, cv2.BORDER_DEFAULT)
# img = cv2.medianBlur(img, 15)  # 滤波函数 解决多余的杂点
# ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# img = cv2.medianBlur(img, 15)
# cv2.THRESH_BINARY：二值化阈值处理会将原始图像处理为仅有两个值的二值图像，
# 其针对像素点的处理方式为：在8位图像中，
# 最大值是255。因此，在对8位灰度图像进行二值化时，如果将阈值设定为127，
# 那么： ● 所有大于127的像素点会被处理为255。 ● 其余值会被处理为0。
# ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)  # 阈值处理
# img = cv2.GaussianBlur(img, (3,3), 0, 0, cv2.BORDER_DEFAULT)
# img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
# ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
kernelR = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernelR)  # 进行开运算
# img = cv2.erode(img, kernelY, (-1, -1), iterations=1)
# img = cv2.dilate(img, kernel, (-1, -1), iterations=1)
# img = cv2.erode(img, kernelR, (-1, -1), iterations=1)
# img = cv2.dilate(img, kernelY, (-1, -1), iterations=1)
# img = cv2.Canny(img, 300, 200, 2)
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernelR)
# img = cv2.dilate(img, kernelY, (-1, -1), iterations=1)
# cv2.find
img_show(img)