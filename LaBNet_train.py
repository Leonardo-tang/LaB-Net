import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from models.BBSNet_model import BBSNet
from models.smt import SMT
from models.BBSNet_SMT_model import BBSNet_SMT
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, EdgeAwareCombinedLoss  # 导入新的损失函数
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt

# 删除原有的IoULoss和CombinedLoss定义，使用utils.py中的EdgeAwareCombinedLoss

# set the device for training
if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
cudnn.benchmark = True

# build the model
model = BBSNet_SMT()
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

if opt.smt_pretrained and os.path.exists(opt.smt_pretrained):
    model.load_smt_pretrained(opt.smt_pretrained)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# 设置损失函数 - 使用边缘感知组合损失
if hasattr(opt, 'edge_weight'):
    edge_weight = opt.edge_weight
else:
    edge_weight = 0.05  # 默认边缘损失权重

criterion = EdgeAwareCombinedLoss(
    edge_weight=edge_weight,
    iou_weight=0.5,
    weights=[0.4, 0.3, 0.2, 0.1]
)

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("BBSNet-SMT with Edge-Aware Loss - Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{};edge_weight:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch, edge_weight))

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0
best_f1 = 0
best_iou = 0


# 验证预训练权重是否加载成功
def check_weights_loaded(model):
    """检查预训练权重是否成功加载"""
    print("\n✅ Checking weights loading status...")

    # 检查SMT模块的参数
    rgb_params = sum(p.numel() for p in model.smt.parameters())
    depth_params = sum(p.numel() for p in model.smt_depth.parameters())

    print(f"RGB SMT parameters: {rgb_params:,}")
    print(f"Depth SMT parameters: {depth_params:,}")

    # 检查双向融合权重
    if hasattr(model, 'eds_bi'):
        print(f"EDS Bi-directional Fusion weights: {model.eds_bi.weight}")
    if hasattr(model, 'sca1_bi'):
        print(f"SCA1 Bi-directional Fusion weights: {model.sca1_bi.weight}")
    if hasattr(model, 'sca2_bi'):
        print(f"SCA2 Bi-directional Fusion weights: {model.sca2_bi.weight}")
    if hasattr(model, 'gfm_bi'):
        print(f"GFM Bi-directional Fusion weights: {model.gfm_bi.weight}")


# 在模型构建后调用
check_weights_loaded(model)


# 计算边缘可视化
def visualize_edges(images, writer, step, prefix=''):
    """可视化边缘图"""
    with torch.no_grad():
        # 确保图像有正确的维度
        if images.dim() == 3:
            images = images.unsqueeze(1)

        # 计算拉普拉斯边缘
        laplace_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)

        # 使用反射填充
        padded = F.pad(images, (1, 1, 1, 1), mode='reflect')
        edges = F.conv2d(padded, laplace_kernel)
        edges = torch.abs(edges)

        # 归一化
        edges = (edges - edges.min()) / (edges.max() - edges.min() + 1e-6)

        # 写入TensorBoard
        if edges.shape[0] > 0:
            grid_image = make_grid(edges[0:1].clone().cpu().data, 1, normalize=True)
            writer.add_image(f'{prefix}Edges', grid_image, step)


# 计算F1和IoU指标
def calculate_metrics(pred, gt):
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)

    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    iou = intersection / (union + 1e-8)

    return f1, iou


# 增强的train函数（添加边缘损失监控）
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    # 记录各损失分量
    loss1_all, loss2_all, loss3_all, loss4_all = 0, 0, 0, 0

    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda().repeat(1, 3, 1, 1)  # 单通道转3通道

            # 模型前向传播
            s1, s2, s3, s4 = model(images, depths)

            # 使用边缘感知组合损失
            loss, loss1, loss2, loss3, loss4 = criterion((s1, s2, s3, s4), gts)

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data

            # 累积各损失分量
            loss1_all += loss1.data
            loss2_all += loss2.data
            loss3_all += loss3.data
            loss4_all += loss4.data

            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           loss.data, loss1.data, loss2.data, loss3.data, loss4.data))

                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                             format(epoch, opt.epoch, i, total_step, loss.data))

                # TensorBoard记录
                writer.add_scalar('Loss/total', loss.data, global_step=step)
                writer.add_scalar('Loss/s1', loss1.data, global_step=step)
                writer.add_scalar('Loss/s2', loss2.data, global_step=step)
                writer.add_scalar('Loss/s3', loss3.data, global_step=step)
                writer.add_scalar('Loss/s4', loss4.data, global_step=step)

                # 可视化
                if i % 500 == 0:
                    grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                    writer.add_image('RGB', grid_image, step)

                    grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                    writer.add_image('Ground_truth', grid_image, step)

                    # 可视化边缘
                    visualize_edges(gts[0:1], writer, step, 'GT_')
                    visualize_edges(torch.sigmoid(s4[0:1]), writer, step, 'Pred_')

                    # 可视化各尺度预测
                    for idx, (name, pred) in enumerate([('s1', s1), ('s2', s2), ('s3', s3), ('s4', s4)]):
                        res = pred[0].clone()
                        res = res.sigmoid().data.cpu().numpy().squeeze()
                        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                        writer.add_image(name, torch.tensor(res), step, dataformats='HW')

        # 计算平均损失
        loss_all /= epoch_step
        loss1_all /= epoch_step
        loss2_all /= epoch_step
        loss3_all /= epoch_step
        loss4_all /= epoch_step

        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}, L1: {:.4f}, L2: {:.4f}, L3: {:.4f}, L4: {:.4f}'.
                     format(epoch, opt.epoch, loss_all, loss1_all, loss2_all, loss3_all, loss4_all))

        writer.add_scalar('Loss-epoch/total', loss_all, global_step=epoch)
        writer.add_scalar('Loss-epoch/s1', loss1_all, global_step=epoch)
        writer.add_scalar('Loss-epoch/s2', loss2_all, global_step=epoch)
        writer.add_scalar('Loss-epoch/s3', loss3_all, global_step=epoch)
        writer.add_scalar('Loss-epoch/s4', loss4_all, global_step=epoch)

        # 每5个epoch保存一次模型
        if epoch % 5 == 0:
            torch.save(model.state_dict(), save_path + f'BBSNet_SMT_epoch_{epoch}.pth')
            print(f'Model saved: BBSNet_SMT_epoch_{epoch}.pth')

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + f'BBSNet_SMT_interrupt_epoch_{epoch}.pth')
        print('save checkpoints successfully!')
        raise


# 增强的test函数
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch, best_f1, best_iou
    model.eval()

    with torch.no_grad():
        mae_sum = 0
        f1_sum = 0
        iou_sum = 0

        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt_np = np.asarray(gt, np.float32)
            gt_np /= (gt_np.max() + 1e-8)

            image = image.cuda()
            depth = depth.cuda().repeat(1, 3, 1, 1)  # 单通道转3通道

            # 模型前向传播
            s1, s2, s3, s4 = model(image, depth)

            # 使用最终输出s4进行测试
            res = s4
            res = F.interpolate(res, size=gt_np.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            # 计算指标
            mae_sum += np.sum(np.abs(res - gt_np)) * 1.0 / (gt_np.shape[0] * gt_np.shape[1])

            # 计算F1和IoU
            f1, iou = calculate_metrics(res, gt_np)
            f1_sum += f1
            iou_sum += iou

        # 计算平均指标
        mae = mae_sum / test_loader.size
        f1 = f1_sum / test_loader.size
        iou = iou_sum / test_loader.size

        # TensorBoard记录
        writer.add_scalar('Test/MAE', mae, global_step=epoch)
        writer.add_scalar('Test/F1', f1, global_step=epoch)
        writer.add_scalar('Test/IoU', iou, global_step=epoch)

        print(f'Epoch: {epoch} MAE: {mae:.4f}, F1: {f1:.4f}, IoU: {iou:.4f}')
        print(f'Best MAE: {best_mae:.4f} at epoch {best_epoch}')
        print(f'Best F1: {best_f1:.4f}, Best IoU: {best_iou:.4f}')

        logging.info(f'#TEST#:Epoch:{epoch} MAE:{mae:.4f} F1:{f1:.4f} IoU:{iou:.4f}')

        # 更新最佳指标
        update_best = False

        if mae < best_mae:
            best_mae = mae
            update_best = True

        if f1 > best_f1:
            best_f1 = f1

        if iou > best_iou:
            best_iou = iou

        if update_best:
            best_epoch = epoch
            torch.save(model.state_dict(), save_path + 'BBSNet_SMT_best.pth')
            print(f'Best model saved at epoch {epoch} with MAE: {mae:.4f}')
            logging.info(f'#BEST#:Epoch:{epoch} MAE:{mae:.4f} F1:{f1:.4f} IoU:{iou:.4f}')

        logging.info(
            f'#BEST_SUMMARY#:BestEpoch:{best_epoch} BestMAE:{best_mae:.4f} BestF1:{best_f1:.4f} BestIoU:{best_iou:.4f}')


if __name__ == '__main__':
    print("=" * 50)
    print("BBSNet-SMT with Edge-Aware Loss Training")
    print("=" * 50)
    print(f"Edge loss weight: {edge_weight}")
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(train_loader) * opt.batchsize}")
    print(f"Test samples: {test_loader.size}")
    print("=" * 50)

    # 记录训练开始时间
    start_time = datetime.now()

    for epoch in range(1, opt.epoch + 1):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch}/{opt.epoch}")
        print(f"{'=' * 50}")

        # 调整学习率
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('LR', cur_lr, global_step=epoch)

        # 训练
        train(train_loader, model, optimizer, epoch, save_path)

        # 测试
        if epoch % 2 == 0:  # 每2个epoch测试一次
            test(test_loader, model, epoch, save_path)

    # 训练结束
    end_time = datetime.now()
    training_time = end_time - start_time

    print("\n" + "=" * 50)
    print("Training Completed!")
    print(f"Total training time: {training_time}")
    print(f"Best epoch: {best_epoch} with MAE: {best_mae:.4f}")
    print(f"Best F1: {best_f1:.4f}, Best IoU: {best_iou:.4f}")
    print("=" * 50)

    # 保存最终模型
    torch.save(model.state_dict(), save_path + 'BBSNet_SMT_final.pth')
    print(f"Final model saved: {save_path}BBSNet_SMT_final.pth")

    writer.close()