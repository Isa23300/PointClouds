#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main_cls.py
@Time: 2018/10/13 10:39 PM

Modified by 
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@Time: 2019/12/30 9:32 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data import ModelNet40,MyShape
from model import PointNet, DGCNN_cls
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import datetime
import logging

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp main_cls.py outputs'+'/'+args.exp_name+'/'+'main_cls.py.backup')
    os.system('cp model.py outputs' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py outputs' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, logger):
    DATA_PATH = './data/modelnet40_normal_resampled'
    train_loader = DataLoader(MyShape(root=DATA_PATH, npoint=1024, split='train'), num_workers=2,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(MyShape(root=DATA_PATH, npoint=1024, split='test'), num_workers=2,
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
    elif args.scheduler == 'step':
        scheduler = StepLR(opt, step_size=20, gamma=0.7)
    
    criterion = cal_loss

    best_test_acc = 0
    best_test_f1 = 0
    best_test_recall = 0
    best_test_precision = 0
    best_test_epoch=0
    best_ytrue = []
    best_ypred = []
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label.long())
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        if args.scheduler == 'cos':
            scheduler.step()
        elif args.scheduler == 'step':
            if opt.param_groups[0]['lr'] > 1e-5:
                scheduler.step()
            if opt.param_groups[0]['lr'] < 1e-5:
                for param_group in opt.param_groups:
                    param_group['lr'] = 1e-5

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,train_loss*1.0/count,
            metrics.accuracy_score(train_true, train_pred),metrics.balanced_accuracy_score(train_true, train_pred))
        logger.info(outstr)
        print(outstr)

        outstr = 'Train %d, f1: %.6f, recall: %.6f, precision: %.6f' % (epoch,metrics.f1_score(train_true, train_pred),
            metrics.recall_score(train_true, train_pred),metrics.precision_score(train_true, train_pred))
        logger.info(outstr)
        print(outstr)
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label.long())
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        test_f1 = metrics.f1_score(test_true, test_pred)
        test_recall = metrics.recall_score(test_true, test_pred)
        test_precision = metrics.precision_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,test_loss*1.0/count,test_acc,avg_per_class_acc)
        logger.info(outstr)
        print(outstr)

        outstr = 'Test %d, f1: %.6f, recall: %.6f, precision: %.6f' % (epoch, test_f1,test_recall,test_precision)
        logger.info(outstr)
        print(outstr)
        if test_acc >= best_test_acc and epoch>10:
            best_test_epoch = epoch
            best_test_acc = test_acc
            best_test_f1 = test_f1
            best_test_recall = test_recall
            best_test_precision = test_precision
            best_ytrue = test_true
            best_ypred = test_pred
            torch.save(model.state_dict(), 'outputs/%s/models/model.t7' % args.exp_name)

        if epoch == args.epochs-1:
            logger.info("y_true:"+str(test_true))
            logger.info("y_pred:"+str(test_pred))
            print("y_true:"+str(test_true))
            print("y_pred:"+str(test_pred))
            outstr = 'Best %d: acc: %.6f, f1: %.6f, recall: %.6f, precision: %.6f' % (best_test_epoch,best_test_acc, best_test_f1,best_test_recall,best_test_precision)
            logger.info(outstr)
            print(outstr)
            logger.info("best y_true:"+str(best_ytrue))
            print("best y_true:"+str(best_ytrue))
            logger.info("best y_pred:"+str(best_ypred))
            print("best y_pred:"+str(best_ypred))


def test(args, logger):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN_cls(args).to(device)
    else:
        raise Exception("Not implemented")

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    logger.info(outstr)
    print(outstr)


def parse_args(x):
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str,default='test' + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')), metavar='N',help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N', choices=['pointnet', 'dgcnn'],help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N', choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=4, metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N', help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Ture:UseSGD False:Use Adam')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N', choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False, help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=x, help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.8, help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    return parser.parse_args()

def main(args):

    '''CREATE DIR'''
    #timestr = 'stepsize_'+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./outputs/')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        log_dir = exp_dir.joinpath(args.exp_name)
    else:
        log_dir = exp_dir.joinpath(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    #file_handler = logging.FileHandler('run.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #logger.info('PARAMETER ...')
    logger.info(args)
    print(args)


    #io = IOStream('outputs/' + args.exp_name + '/run.log')
    #io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        logger.info(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
        print('Using GPU')
    else:
        logger.info('Using CPU')
        print('Using CPU')

    if not args.eval:
        train(args, logger)
    else:
        test(args, logger)


if __name__ == "__main__":
    # Training settings
    for x in [2048]:
        args = parse_args(x)
        _init_()
        main(args)