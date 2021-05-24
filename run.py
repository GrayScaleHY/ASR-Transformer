import os
import yaml
import numpy as np
import random
import torch
import argparse
import time
import shutil
from otrans.model.transformer import Transformer
from otrans.optim import TransformerOptimizer
from otrans.train import Trainer
from otrans.data import FeatureLoader


def main(args):

    ## 使下一次的随机数固定
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    ## 加载参数
    with open(args.config, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    ## 保存模型
    expdir = os.path.join(params['model']['expdir'],time.strftime("%m-%d-%H-%M-%S"))
    if not os.path.exists(expdir):
        os.makedirs(expdir)
    shutil.copy(params['data']['vocab'],os.path.join(expdir,"vocab"))
    shutil.copy(args.config,os.path.join(expdir,"config.yaml"))
    
    ## 加载模型
    model = Transformer(params['model'])
    if args.ngpu >= 1:
        model.cuda()
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)

    # build optimizer
    optimizer = TransformerOptimizer(model, params['train'], model_size=params['model']['d_model'],
                                     parallel_mode=args.parallel_mode)

    trainer = Trainer(params, model=model, optimizer=optimizer, is_visual=True, expdir=expdir, ngpu=args.ngpu,
                      parallel_mode=args.parallel_mode, local_rank=args.local_rank)

    train_loader = FeatureLoader(
        params, 'train', shuffle=params['train']['shuffle'],
        ngpu=args.ngpu, mode=args.parallel_mode)
    trainer.train(train_loader=train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="config/transformer.yaml")
    parser.add_argument('-n', '--ngpu', type=int, default=4)
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('-p', '--parallel_mode', type=str, default='dp')
    parser.add_argument('-r', '--local_rank', type=int, default=0)
    cmd_args = parser.parse_args()
    main(cmd_args)
