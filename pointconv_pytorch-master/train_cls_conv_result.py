import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_simple_checkpoint
from model.pointconv import PointConvDensityClsSsg as PointConvClsSsg
import provider
import numpy as np
import sklearn.metrics as metrics


def parse_args(x):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=4, help='batch size in training')
    parser.add_argument('--epoch',  default=350, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.000001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=x, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=2, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--gamma', type=float, default=0.6, help='StepLR,default=0.7')
    parser.add_argument('--stepsize',type=int, default=30, help='StepLR,default=30')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/dome_'+str(args.num_point)+'_'+str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = './data/modelnet40_normal_resampled/'

    TRAIN_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='train', normal_channel=args.normal)
    TEST_DATASET = ModelNetDataLoader(root=DATA_PATH, npoint=args.num_point, split='test', normal_channel=args.normal)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers)

    logger.info("The number of training data is: %d", len(TRAIN_DATASET))
    logger.info("The number of test data is: %d", len(TEST_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    num_class = 2
    classifier = PointConvClsSsg(num_class).cuda()
    print(str(classifier))
    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        checkpoint = torch.load(args.pretrain)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    best_y_t=[]
    best_y_p=[]
    best_epoch=0
    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)

        scheduler.step()
        train_pred = []
        train_true = []
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points = points.data.numpy()
            # jittered_data = provider.random_scale_point_cloud(points[:,:, 0:3], scale_low=2.0/3, scale_high=3/2.0)
            # jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
            # points[:, :, 0:3] = jittered_data
            # points = provider.random_point_dropout_v2(points)
            provider.shuffle_points(points)
            points = torch.Tensor(points)
            target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred = classifier(points[:, :3, :], points[:, 3:, :])
            loss = F.nll_loss(pred, target.long())
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
        #print('Train Accuracy: %f' % train_acc)
        #logger.info('Train Accuracy: %f' % train_acc)
#_____________test____________________________________________

        with torch.no_grad():
            test_pred = []
            test_true = []
            acc = 0
            if epoch > 10:
                for j, data in enumerate(testDataLoader, 0):
                    points, target = data
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.cuda(), target.cuda()
                    classifier = classifier.eval()
                    with torch.no_grad():
                        pred = classifier(points[:, :3, :], points[:, 3:, :])
                    pred_choice = pred.data.max(1)[1]

                    test_true.append(target.cpu().numpy())
                    test_pred.append(pred_choice.detach().cpu().numpy())

                test_true = np.concatenate(test_true)
                test_pred = np.concatenate(test_pred)

                acc = metrics.accuracy_score(test_true, test_pred)
                outstr = 'Test %d, test acc: %.6f, test avg acc: %.6f' % (epoch,acc,metrics.balanced_accuracy_score(test_true, test_pred))
                logger.info(outstr)
                print(outstr)

                outstr = 'Test %d, f1: %.6f, recall: %.6f, precision: %.6f' % (epoch, metrics.f1_score(test_true, test_pred),
                        metrics.recall_score(test_true, test_pred),metrics.precision_score(test_true, test_pred))
                logger.info(outstr)
                print(outstr)
#____________________________________________________________________________
        if (acc >= best_tst_accuracy) and epoch > 10:
            best_epoch = epoch
            best_tst_accuracy = acc
            best_y_t = test_true
            best_y_p = test_pred
            #logger.info('Save model...')
            save_simple_checkpoint(
                global_epoch + 1,
                train_acc,
                acc,
                classifier,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')

        if epoch==args.epoch-1:
            logger.info('epoch '+str(best_epoch)+': best_acc: '+str(best_tst_accuracy))
            logger.info('best_y_true'+str(best_y_t))
            logger.info('best_y_pred'+str(best_y_p))

        #print('\r Loss: %f' % loss.data)
        #logger.info('Loss: %f', loss.data)
        #print('\r Test %s: %f   ***  %s: %f' % (blue('Accuracy'),acc, blue('Best Accuracy'),best_tst_accuracy))
        #logger.info('Test Accuracy: %f  *** Best Test Accuracy: %f', acc, best_tst_accuracy)


        global_epoch += 1
    print('Best Accuracy: %f'%best_tst_accuracy)

    logger.info('End of training...')

if __name__ == '__main__':
    list_num=[2048]
    for x in list_num:
        args = parse_args(x)
        main(args)