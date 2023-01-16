import os

import utils
from PIL import Image
import predic

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' ) # 定义device对象
# test_demo = MyDataset(utils.add_file_to_list('D:/soft/python/python_project/lenet/test'), utils.transform)
file_path = 'D:/soft/python/python_project/lenet/test'
part = os.listdir(file_path)
label_dict = {}
for label in part:
    img_list_path = file_path + '/' + label
    if not utils.start01(img_list_path):
        continue
    img_list = os.listdir(img_list_path)
    for loc, img_name in enumerate(img_list):
        img_path = img_list_path + '/' + img_name
        img_list[loc] = img_path
    label_dict[label] = img_list

total_count = 0
fail_count = 0
for key in label_dict.keys():
    for img_path in label_dict[key]:
        total_count += 1
        img = Image.open(img_path)
        char_name = predic.predic_char(img)
        char_name = char_name if len(key) == 1 else utils.privance[key]
        if char_name is not key:
            fail_count += 1
            print('预测失败 %s, 预测的结果是 %s,本来的结果是 %s' % (img_path, char_name, key))

print('运行结束')
print('准确率为', (1 - fail_count / total_count))
