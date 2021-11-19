import os
import torch
import torch.backends.cudnn as cudnn
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from utils import getImgList, PriorBox, decode, py_cpu_nms
from models.retinaFace import RetinaFace
from config import cfg_re50, cfg_mnet


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    weight = 'weights/retina_face/Resnet50_Final.pth'
    backbone = 'resnet50'  # or 'mobile0.25'
    cpu = True  # or False to use cuda
    confidence_threshold = 0.05
    nms_threshold = 0.55
    top_k = 5000
    keep_top_k = 500

    # load all images
    # imgDir = 'sourceFiles/profile pics'
    # imgList = getImgList(imgDir)
    # or load image only in  'User demo profiles.json'
    imgDir = 'sourceFiles'
    with open("./sourceFiles/User demo profiles.json", 'r', encoding='UTF-8') as load_f2:
        profile_dict = json.load(load_f2)
    imgList = []
    for i, item in enumerate(profile_dict):
        imgList.append(item["img_path"])

    cfg = None
    if backbone == "mobile0.25":
        cfg = cfg_mnet
    elif backbone == "resnet50":
        cfg = cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, weight, cpu)
    net.eval()
    print('Finished loading model')
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    resize = 1

    show_image = False
    # show_image = True  # for test
    # inference(weight, backbone, os.path.join(imgDir, imgList[0]))
    is_face = []
    coefficients = []
    for idx, imgPath in enumerate(tqdm(imgList)):
        # if idx > 99:
        #     break
        img_raw = cv2.imread(os.path.join(imgDir, imgPath))
        if img_raw is None:
            coefficients.append(-1)
            print(imgPath + " is not existed")
            continue
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        loc, conf, _ = net(img)  # forward pass
        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]

        # keep top-K faster NMS
        # dets = dets[:keep_top_k, :]
        dets = dets[:1, :]  # for is_face or not, one is enough
        # if len(dets[0]) < 3:
        #     is_face.append(0)
        #     coefficients.append(0.)
        #     continue
        if len(dets) == 0 or len(dets[0]) < 3 or dets[0][4] < nms_threshold:
            # is_face.append(0)
            coefficients.append(0.)
            continue
        else:
            # is_face.append(1)
            score = "{:.4f}".format(dets[0][4])
            coefficients.append(score)

        if show_image:
            for b in dets:
                if b[4] < nms_threshold:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.imshow(str(imgPath), img_raw)
            cv2.waitKey(500)

    dataFrame = pd.DataFrame({'img': imgList, 'coefficient': coefficients})
    dataFrame.to_csv('results/result_User_demo_profiles_json.csv', index=False)
