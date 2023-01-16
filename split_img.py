from PIL import Image
import cv2
import numpy as np


def img_show(img):
    Image.Image.show(Image.fromarray(img))


def open_handle(img):
    # 对图片进行开运算, 腐蚀和膨胀操作
    kernel_X = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))           # 定义矩形卷积核
    mark = cv2.dilate(img, kernel_X, (-1, -1),iterations=2)                # 膨胀操作
    mark = cv2.erode(mark, kernel_X, (-1, -1), iterations=4)
    return mark

def erode_img(img, iteration_num=1):
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.erode(img, kernel, (-1, -1), iterations=iteration_num)

def img_dilate(img, times=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    cv2.dilate(img, kernel, (-1, -1), iterations=times)



def p2(img):
    gray_img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (3,3), 0, 0, cv2.BORDER_DEFAULT)
    mark3 = gray_img_
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20,1))
    # mark3 = cv2.morphologyEx(mark3, cv2.MORPH_CLOSE, kernel)
    mark3 = cv2.dilate(mark3, kernel, (-1,-1), iterations=3)
    mark3 = cv2.Canny(mark3, 100, 200, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    mark3 = cv2.morphologyEx(mark3, cv2.MORPH_CLOSE, kernel)
    img_show(mark3)

def predict(img):
    """
    这个函数通过一系列的处理，找到可能是车牌的一些矩形区域
    输入： imageArr是原始图像的数字矩阵
    输出：gray_img_原始图像经过高斯平滑后的二值图
          contours是找到的多个轮廓
    """
    imageArr = cv2.imread(img)
    img_copy = imageArr.copy()
    gray_img = cv2.cvtColor(img_copy , cv2.COLOR_BGR2GRAY)
    gray_img_ = cv2.GaussianBlur(gray_img, (5,5), 0, 0, cv2.BORDER_DEFAULT)
    img_show(gray_img_)
    kernel = np.ones((23, 23), np.uint8)
    img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    img_show(img_opening)
    img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0)
    img_show(img_opening)
    # 找到图像边缘
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # # 使用开运算和闭运算让图像边缘成为一个整体
    kernel = np.ones((10, 10), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    img_show(img_edge2)
    # # 查找图像边缘整体形成的矩形区域，可能有很多，车牌就在其中一个矩形区域中
    image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return gray_img_, contours


if __name__ == '__main__':
    img_path = 'D:\soft\python\python_project\lenet\img_file\img_2.png'
    res = predict(img_path)
    # Image.Image.show(Image.fromarray(res))
    p2(cv2.imread(img_path))