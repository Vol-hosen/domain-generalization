from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default='logs/CUHK-PEDES/iira/configs.yaml')
    args = parser.parse_args()
    #dataset_name = args.dataset_name if 'dataset_name' in args else ''
    args = load_train_configs(args.config_file)
    #args.dataset_name = dataset_name if dataset_name != '' else args.dataset_name

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    except_keys = ['classifier.weight', 'classifier.bias']
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'), except_keys=except_keys)
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)
    #python test.py --config_file 'logs/CUHK-PEDES/20260112_201928_irra-cuhk-mse-v2/configs.yaml'
    #python test.py --config_file 'logs/CUHK-PEDES/20260112_173820_irra-cuhk/configs.yaml'
    #python test.py --config_file 'logs/ICFG-PEDES/20260113_173539_irra-icfg-kl-v2/configs.yaml'