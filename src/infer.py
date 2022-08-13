# coding: utf-8

import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import argparse
from model import module


def infer_sketch_gan(opt):
    img = Image.open( opt.input ).convert('L')
    w, h  = img.size[0], img.size[1]
    pw    = 8-(w%8) if w%8!=0 else 0
    ph    = 8-(h%8) if h%8!=0 else 0
    immean, imstd, model_path, remote_model = get_default_args(opt.model)
    gpu = opt.gpu=="1" and torch.cuda.is_available()
    data  = ((transforms.ToTensor()(img)-immean)/imstd).unsqueeze(0)
    if pw!=0 or ph!=0:
        data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data
    data = data.float().cuda() if gpu else data.float()

    model = module.Net()
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.hub.load_state_dict_from_url(remote_model, progress=True)
    model.load_state_dict(checkpoint)
    model.eval()

    save_img_path = opt.output
    if not save_img_path:
        save_img_path = opt.input + f".out.{opt.model}.jpg"
    with torch.no_grad():
        pred = model.cuda().forward( data ).float() if gpu else model.forward( data ).float()
        save_image(pred, save_img_path)


mean_std = {
    "gan": (0.9664114577640158, 0.0858381272736797, "../models/model_gan.pth", "https://github.com/aidreamwin/sketch_simplification_pytorch/releases/download/model/model_gan.pth"),
    "mse": (0.9664423107454593, 0.08583666033640507, "../models/model_mse.pth", None),
    "pencil_artist1": (0.9817833515894078, 0.0925009022585048, "../models/model_pencil_artist1.pth", None),
    "pencil_artist2": (0.9851298627337799, 0.07418377454883571, "../models/model_pencil_artist2.pth", None),
}

def get_default_args(model_name):
    return mean_std.get(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../image/1.jpg', help='path of the input image')
    parser.add_argument('--output', type=str, default='', help='path of the out image')
    parser.add_argument('--gpu', type=str, default='0',choices=["0", "1"], help='use gpu')
    parser.add_argument('--model', type=str, default='gan',choices=["gan", "mse", "pencil_artist1", "pencil_artist2"], help='name of the model')
    opt = parser.parse_args()
    infer_sketch_gan(opt)