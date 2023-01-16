import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet, Net
import utils

def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = utils.categories

    # net = LeNet()
    # net.load_state_dict(torch.load('./dataset/模型1.pth'))

    # path = utils.change_file_type('D:\soft\python\python_project\lenet\dataset\qwe\4\31-4.jpg')
    im = Image.open('D:/soft/python/python_project/lenet/3.jpg')
    # im = utils.transform(im)  # [C, H, W]
    # im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    print(predic_char(im))

    # with torch.no_grad():
    #     outputs = net(im)
    #     predict = torch.max(outputs, dim=1)[1].numpy()
    #
    # print(int(predict))
    # print(classes[int(predict)])

def predic_char(img):
    classes = utils.categories

    net = LeNet()
    net.load_state_dict(torch.load('./Leng1.pth', map_location=torch.device('cpu')))

    # path = utils.change_file_type('D:\soft\python\python_project\lenet\dataset\qwe\4\31-4.jpg')
    im = img
    im = utils.transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()

    # print(int(predict))
    # print(classes[int(predict)])
    return classes[int(predict)]


if __name__ == '__main__':
    main()