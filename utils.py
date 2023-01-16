import os

import torch
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.Resize(size=(32, 32)),
     transforms.Grayscale(3),
     transforms.PILToTensor(),
     transforms.ConvertImageDtype(torch.float),
     # transforms.Normalize(mean=[127.5], std=[127.5])
     ]
)


def change_file_type(file_path):
    a = file_path.split('\\')
    new_path = ''
    for i in range(len(a)):
        new_path += a[i]
        new_path += '/'
    return new_path


def add_file_to_list(file_path):
    # file_path = change_file_type(file_path)
    ls = os.listdir(file_path)
    label_dict = {}
    for i in range(len(ls)):
        file_name = ls[i]
        ls[i] = file_path + '/' + ls[i]
        if not os.path.isdir(ls[i]):
            ls[i] = ''
            continue
        label_dict[file_name] = ls[i]
    return label_dict


print(change_file_type('D:\soft\python\python_project\lenet\dataset\qwe'))

_provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
              "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新","湖"]
_ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
        "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

categories = []
categories.extend(_ads)
categories.extend(_provinces)
total_cate = {'A': (0, 'A'), 'B': (1, 'B'), 'C': (2, 'C'), 'D': (3, 'D'), 'E': (4, 'E'),
              'F': (5, 'F'), 'G': (6, 'G'), 'H': (7, 'H'), 'J': (8, 'J'), 'K': (9, 'K'),
              'L': (10, 'L'), 'M': (11, 'M'), 'N': (12, 'N'), 'P': (13, 'P'), 'Q': (14, 'Q'),
              'R': (15, 'R'), 'S': (16, 'S'), 'T': (17, 'T'), 'U': (18, 'U'), 'V': (19, 'V'),
              'W': (20, 'W'), 'X': (21, 'X'), 'Y': (22, 'Y'), 'Z': (23, 'Z'), '0': (24, '0'),
              '1': (25, '1'), '2': (26, '2'), '3': (27, '3'), '4': (28, '4'), '5': (29, '5'),
              '6': (30, '6'), '7': (31, '7'), '8': (32, '8'), '9': (33, '9'),
              '皖': (34, '皖'), '沪': (35, '沪'), '津': (36, '津'), '渝': (37, '渝'),
              '冀': (38, '冀'), '晋': (39, '晋'), '蒙': (40, '蒙'), '辽': (41, '辽'),
              '吉': (42, '吉'), '黑': (43, '黑'), '苏': (44, '苏'), '浙': (45, '浙'),
              '京': (46, '京'), '闽': (47, '闽'), '赣': (48, '赣'), '鲁': (49, '鲁'),
              '豫': (50, '豫'), '鄂': (51, '鄂'), '湘': (52, '湘'), '粤': (53, '粤'),
              '桂': (54, '桂'), '琼': (55, '琼'), '川': (56, '川'), '贵': (57, '贵'),
              '云': (58, '云'), '藏': (59, '藏'), '陕': (60, '陕'), '甘': (61, '甘'),
              '青': (62, '青'), '宁': (63, '宁'), '新': (64, '新'), '湖': (65, '湖')}

privance = {'cuan': '川', 'e1': '鄂', 'gan': '赣', 'gan1': '甘', 'gui': '贵', 'gui1': '桂', 'hei': '黑',
            'hu': '湖', 'ji': '冀', 'jin': '津', 'jing': '京', 'jl': '吉', 'liao': '辽', 'lu': '鲁',
            'meng': '蒙', 'min': '闽', 'ning': '宁', 'qing': '青', 'qiong': '琼', 'shan': '陕', 'su': '苏',
            'sx': '晋', 'wan': '皖', 'xiang': '湘', 'xin': '新', 'yu': '豫', 'yu1': '渝', 'yue': '粤', 'yun': '云',
            'zang': '藏', 'zhe': '浙'}


# print(len(privance))
# print(len(categories))
# for i in range(len(categories)):
#     # print('\'', categories[i], '\'', ':', '(', i, ',', categories[i], ')', end=',')
#     print('\'%s\':(%d,\'%s\')' % (categories[i], i, categories[i]), end=',')

def start01(str1):
    var = len(str1) > 1 and (str1[0] != '_')
    return var
# a1 = os.listdir('D:/soft/python/python_project/lenet/dataset/qwe')
# for b1 in a1:
#     if start01(b1) and b1 not in privance.keys():
#         print(b1)