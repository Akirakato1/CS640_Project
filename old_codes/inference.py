import cv2
import numpy as np
import torch
from tqdm import tqdm
import os

from model import get_model
from utils import getImgList


# de
@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight,map_location=torch.device('cpu')))
    net.eval()
    feat = net(img).numpy()
    print(feat)


if __name__ == "__main__":
    weight = 'weights/ms1mv3_arcface_r50_fp16/backbone.pth'
    backbone = 'r50'
    imgDir = 'sourceFiles/profile pics'
    imgList = getImgList(imgDir)
    inference(weight, backbone, os.path.join(imgDir, imgList[0]))

    # net = get_model(backbone, fp16=False)
    # net.load_state_dict(torch.load(weight))
    # net.eval()
    #
    # with torch.no_grad():
    #     for idx, imgPath in enumerate(tqdm(imgList)):
    #         img = cv2.imread(os.path.join(imgDir,imgPath))
    #         img = cv2.resize(img, (112, 112))
    #         feat = net(img).numpy()
