import torch

import argparse


def parse_option():

    parser = argparse.ArgumentParser('argument for dino testing')

    parser.add_argument("--dino_type", type=str, default="dino_vits16",
                        choices=['dino_vits16', 'dino_vits8', 'dino_vitb16', 'dino_vitb8', 'dino_resnet50'])

    opt = parser.parse_args()

    return opt

def load_dino(opt):

    model = torch.hub.load('facebookresearch/dino:main', opt.dino_type)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    return model



if __name__ == "__main__":

    opt = parse_option()
    model = load_dino(opt)