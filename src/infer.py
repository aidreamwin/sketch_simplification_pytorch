# coding: utf-8

import os
from sys import flags
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from collections import OrderedDict

from model import sketch_gan

def infer_sketch_gan(file_path: str, gpu=True):
    img = Image.open( file_path ).convert('L')
    w, h  = img.size[0], img.size[1]
    pw    = 8-(w%8) if w%8!=0 else 0
    ph    = 8-(h%8) if h%8!=0 else 0
    immean = 0.9664114577640158
    imstd  = 0.0858381272736797
    gpu = True
    data  = ((transforms.ToTensor()(img)-immean)/imstd).unsqueeze(0) # .float().cuda() if gpu else ((transforms.ToTensor()(img)-immean)/imstd).unsqueeze(0).float()
    if pw!=0 or ph!=0:
        print(pw, ph)
        data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data
    data = data.float().cuda() if gpu else data.float()

    model = sketch_gan.Net()

    checkpoint = torch.load('../models/model_gan.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    with torch.no_grad():
        pred = model.cuda().forward( data ).float() if gpu else model.forward( data ).float()
        basename = os.path.basename(file_path)
        save_image(pred, f"../image/{basename}-out.jpg")


if __name__ == "__main__":
    infer_sketch_gan("../image/1.jpg", True)