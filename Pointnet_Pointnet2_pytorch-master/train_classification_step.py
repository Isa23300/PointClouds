"""
Author: Benny
Date: Nov 2019
"""
import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse

import sklearn.metrics as metrics
from pathlib import Path
from tqdm import tqdm
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
def parse_args(x,y):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--filename', type=str, default='ga_'+str(x)+'_st'+str(y), help='use cpu mode')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_msg', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40') # add
    parser.add_argument('--epoch', default=50, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.00005, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--gamma', type=float, default=x, help='StepLR,default=0.7')
    parser.add_argument('--stepsize', type=int, default=y, help='StepLR,default=30')

    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    #timestr = 'stepsize_'+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    timestr = args.filename
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #logger.info('PARAMETER ...')
    logger.info(args)
    print(args)

    '''DATA LOADING'''
    #logger.info('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('models/pointnet2_utils.py', str(exp_dir))
    shutil.copy('./train_classification.py', str(exp_dir))
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    print(str(classifier))
    '''
    try: # 改路径
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')#重新运行
        #checkpoint = torch.load('log/classification/2023-07-10_11-40/checkpoints/best_model.pth')#继续上次
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
    '''
    #logger.info('No existing model, starting training from scratch...')
    start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_epoch=0
    '''TRANING'''
    #logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()

        scheduler.step()
        train_pred = []
        train_true = []
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = provider.random_point_dropout(points)
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            train_true.append(target.cpu().numpy())
            train_pred.append(pred_choice.detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            global_step += 1

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,loss.data,train_acc,metrics.balanced_accuracy_score(train_true, train_pred))
        logger.info(outstr)
        print(outstr)

        outstr = 'Train %d, f1: %.6f, recall: %.6f, precision: %.6f' % (epoch,metrics.f1_score(train_true, train_pred),
            metrics.recall_score(train_true, train_pred),metrics.precision_score(train_true, train_pred))
        logger.info(outstr)
        print(outstr)

        with torch.no_grad():
            #————————————————————————————————————————test----------------------------
            test_pred = []
            test_true = []
            class_acc = np.zeros((num_class, 3))
            classifier = classifier.eval()
            for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
                if not args.use_cpu:
                    points, target = points.cuda(), target.cuda()

                points = points.transpose(2, 1)
                pred, trans_feat = classifier(points)
                test_loss = F.nll_loss(pred, target.long())
                pred_choice = pred.data.max(1)[1]

                test_true.append(target.cpu().numpy())
                test_pred.append(pred_choice.detach().cpu().numpy())

            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)

            instance_acc = metrics.accuracy_score(test_true, test_pred)

            outstr = 'Test %d, test acc: %.6f, test avg acc: %.6f, loss: %6f' % (
                epoch, instance_acc, metrics.balanced_accuracy_score(test_true, test_pred),test_loss.data)
            logger.info(outstr)
            print(outstr)
            #---------------------------test over----------------------
            if (epoch>10):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/'+str(args.num_point)+'best_model.pth'
                #logger.info('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    #logger.info('End of training...')



if __name__ == '__main__':
    list_gamma=[0.5,0.6,0.7,0.8,0.9]
    list_step=[5,10,15,20,25,30]
    for x in list_gamma:
        for y in list_step:
            args = parse_args(x,y)
            main(args)

