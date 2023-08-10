"""
Author: Benny
Date: Nov 2019
"""
import os
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import torch
import logging
from tqdm import tqdm
import sys
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size in training')
    parser.add_argument('--num_category', default=2, type=int, choices=[2, 10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--pc_path', type=str, default='data/modelnet40_normal_resampled/myshape2/myshape2_0003.txt', help='pc path')
    parser.add_argument('--model', type=str, default='pointnet_cls', help='model name')
    parser.add_argument('--log_dir', type=str, default='log/classification/2023-06-26_06-55', help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting')
    return parser.parse_args()


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def test(model, loader, num_class=40, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    tmp_class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, tmp_class_acc,class_acc


def main(args):
    
    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = args.model
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(args.log_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
     
    cat = [line.rstrip() for line in open('data/modelnet40_normal_resampled/myshape2_names.txt')]

    if args.use_normals: # 用法线则全部加载
        pc = np.loadtxt(args.pc_path,delimiter=',')
    else:
        pc = np.loadtxt(args.pc_path,delimiter=',')[:,:3]

    pc = farthest_point_sample(pc,args.num_point)
    pc[:, 0:3] = pc_normalize(pc[:, 0:3])
    pc = pc[None,...]

    classifier.eval()
    with torch.no_grad():
        pc = torch.tensor(pc,dtype=torch.float32).cuda()
        pc = pc.transpose(2, 1)
        pred, _ = classifier(pc)
        pred_choice = pred.data.max(1)[1]
        print('input: {}, pred: {}'.format(args.pc_path.split('/')[-2],cat[pred_choice]))


if __name__ == '__main__':
    args = parse_args()
    main(args)
