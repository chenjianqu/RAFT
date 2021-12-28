import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

import sys
sys.path.append('core')
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="models/raft-kitti.pth", help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--export',default="kitti.pt", help="export torchscript model")

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to("cuda")
    model.eval()

    trace_model = torch.jit.script(model)
    trace_model.save(args.export)

