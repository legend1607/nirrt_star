import os
import logging
import argparse
import importlib
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from environment.kuka_env import KukaEnv
from pointnet_pointnet2 import point_utils
from pointnet_pointnet2.PathPlanDataLoader import PathPlanDataset

classes = ['other free points', 'optimal path points']
NUM_CLASSES = len(classes)
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {i: cat for i, cat in enumerate(seg_classes.keys())}

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    model_name = args.env+'_'+args.model+'_'+str(args.dim)+'d'
    experiment_dir = os.path.join('results/model_training', model_name)
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints')
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    # logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, model_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # TensorBoard writer
    tb_writer = SummaryWriter(log_dir=os.path.join(experiment_dir, 'tensorboard'))

    BATCH_SIZE = args.batch_size
    env_type = args.env+'_'+str(args.dim)+'d'

    # Datasets and loaders
    TRAIN_DATASET = PathPlanDataset(env_type,dataset_filepath='data/'+env_type+'/train.npz')
    VAL_DATASET = PathPlanDataset(env_type,dataset_filepath='data/'+env_type+'/val.npz')
    trainDataLoader = torch.utils.data.DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=10,
        drop_last=False,
    )
    valDataLoader = torch.utils.data.DataLoader(
        VAL_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=10,
        drop_last=False,
    )
    NUM_POINT = TRAIN_DATASET.n_points
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of validation data is: %d" % len(VAL_DATASET))

    # Model
    MODEL = importlib.import_module('pointnet_pointnet2.models.'+args.model)
    classifier = MODEL.get_model(NUM_CLASSES,in_channels=TRAIN_DATASET.d).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_'+model_name+'.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    ) if args.optimizer=='Adam' else torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_optimal_path_IoU = None

    if args.env=='kuka':
        env=KukaEnv(dim=args.dim)
    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        # 更新学习率
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 更新 BatchNorm momentum
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        momentum = max(momentum, 0.01)
        classifier = classifier.apply(lambda x: setattr(x, 'momentum', momentum) if isinstance(x, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)) else None)

        # Training
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier.train()
        for i, batch in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            pc_xyz_raw, pc_xyz, pc_features, pc_labels, token = batch
            if args.env=='kuka':
                pc_xyz = point_utils.augment_kuka_joint_space(pc_xyz.data.numpy(), env=env)
            else:
                pc_xyz=point_utils.rotate_point_cloud_z(pc_xyz.data.numpy())
            pc_xyz = torch.Tensor(pc_xyz)
            points = torch.cat([pc_xyz, pc_features], dim=2).float().cuda()
            target = pc_labels.long().cuda()
            points = points.transpose(2,1)
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1,1)[:,0].cpu().numpy()
            target_flat = target.view(-1,1)[:,0]
            loss = criterion(seg_pred, target_flat, trans_feat, weights)
            loss.backward()
            optimizer.step()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss.item()
        train_loss = loss_sum / len(trainDataLoader)
        train_acc = total_correct / float(total_seen)
        log_string(f'Train loss: {train_loss}, acc: {train_acc}')
        tb_writer.add_scalar('Train/Loss', train_loss, global_epoch)
        tb_writer.add_scalar('Train/Acc', train_acc, global_epoch)

        # Validation
        classifier.eval()
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        for i, batch in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
            pc_xyz_raw, pc_xyz, pc_features, pc_labels, token = batch
            pc_xyz = torch.Tensor(pc_xyz.data.numpy())
            points = torch.cat([pc_xyz, pc_features], dim=2).float().cuda()
            target = pc_labels.long().cuda()
            points = points.transpose(2,1)
            with torch.no_grad():
                seg_pred, trans_feat = classifier(points)
            seg_pred_np = seg_pred.contiguous().cpu().numpy()
            seg_pred_flat = seg_pred.contiguous().view(-1, NUM_CLASSES)
            batch_label = target.view(-1,1)[:,0].cpu().numpy()
            target_flat = target.view(-1,1)[:,0]
            loss = criterion(seg_pred_flat, target_flat, trans_feat, weights)
            loss_sum += loss.item()
            pred_val = np.argmax(seg_pred_np,2)
            correct = np.sum(pred_val == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            for l in range(NUM_CLASSES):
                total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                total_iou_deno_class[l] += np.sum((pred_val == l) | (batch_label == l))

        val_loss = loss_sum / len(valDataLoader)
        val_acc = total_correct / float(total_seen)
        mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float32)+1e-6))
        log_string(f'Val loss: {val_loss}, acc: {val_acc}, mIoU: {mIoU}')
        tb_writer.add_scalar('Val/Loss', val_loss, global_epoch)
        tb_writer.add_scalar('Val/Acc', val_acc, global_epoch)
        tb_writer.add_scalar('Val/mIoU', mIoU, global_epoch)

        # Save best model
        optimal_path_IoU = total_correct_class[1] / float(total_iou_deno_class[1])
        if best_optimal_path_IoU is None or optimal_path_IoU >= best_optimal_path_IoU:
            best_optimal_path_IoU = optimal_path_IoU
            savepath = os.path.join(checkpoints_dir, 'best_'+model_name+'.pth')
            state = {
                'epoch': epoch,
                'class_avg_iou': mIoU,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saved best model at epoch %d' % epoch)

        global_epoch += 1

    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--dim', type=int, default=3, help='environment dimension: 2 or 3.')
    parser.add_argument('--env', type=str, default='kuka_random', choices=['random', 'kuka_random'])
    parser.add_argument('--model', type=str, default='pointnet2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--lr_decay', type=float, default=0.7)
    parser.add_argument('--random_seed', type=int, default=None)
    args = parser.parse_args()
    main(args)
