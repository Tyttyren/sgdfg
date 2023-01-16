import numpy as np
import cv2
import os

words_list = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]

con_list = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"
]


def order_points(pts):
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype='float32')

    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)  # 每行像素值进行相加；若axis=0，每列像素值相加
    rect[0] = pts[np.argmin(s)]  # top_left,返回s首个最小值索引，eg.[1,0,2,0],返回值为1
    rect[2] = pts[np.argmax(s)]  # bottom_left,返回s首个最大值索引，eg.[1,0,2,0],返回值为2

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)  # 第i+1列减第i列
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点,左上角为原点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取透视变换矩阵并应用它
    M = cv2.getPerspectiveTransform(rect, dst)
    # 进行透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


if __name__ == "__main__":
    global points
    total = []
    for item in os.listdir(os.path.join('/civi/CCPD_Test/CPDD_Dataset/train_dataset/')):
        img = cv2.imread('/civi/CCPD_Test/CPDD_Dataset/train_dataset/' + item)
        _, _, bbox, points, label, _, _ = item.split('-')

        points = points.split('_')
        tmp = points
        points = []
        for _ in tmp:
            points.append([int(_.split('&')[0]), int(_.split('&')[1])])
        # print(points)

        label = label.split('_')
        con = con_list[int(label[0])]
        words = [words_list[int(_)] for _ in label[1:]]
        label = con + ''.join(words)
        line = item + '\t' + label
        line = line[:] + '\n'
        total.append(line)

        # 还原像素位置
        points = np.array(points, dtype=np.float32)
        warped = four_point_transform(img, points)
        save_path = os.path.join('/civi/CCPD_Test/challenge_dataset/train_dataset', item)
        cv2.imwrite(save_path, warped)

    with open('/civi/CCPD_Test/challenge_dataset/' + 'train_dataset.txt', 'w', encoding='UTF-8') as f:
        for line in total:
            f.write(line)
