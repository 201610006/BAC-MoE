import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from medpy.metric.binary import assd, hd95
from sklearn.metrics import f1_score
from my_dataset import MyDataset
from BACMoE_Model import BACMoE_Model
import torch.nn.functional as F
import pandas as pd
import cv2
from matplotlib import cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)
# ========================== #
#    多类别评估函数定义      #
# ========================== #

names_to_abb = {'丘脑': 'TH', '中脑': 'MB', '中脑导水管': 'CA', '侧脑室前角': 'AHLV', '侧脑室后角': 'PHLV', '大脑外侧裂': 'LS',
                   '大脑实质': 'CP', '大脑镰': 'FC', '小脑半球': 'CBH', '小脑蚓部': 'CBV', '穹窿柱': 'CF', '第三脑室': '3V', '胼胝体': 'CC',
                   '脉络丛': 'CPX', '脑岛': 'IN', '透明隔': 'CSP', '透明隔腔': 'SP', '颅后窝池': 'CM', '颅骨光环': 'SH'}

def evaluate_multiclass_segmentation(preds, gts, class_names):
    num_classes = len(class_names)
    dice_scores = [[] for _ in range(num_classes)]
    assd_scores = [[] for _ in range(num_classes)]
    hd_scores = [[] for _ in range(num_classes)]

    for pred, gt in zip(preds, gts):
        for c in range(num_classes):
            pred_c = pred[c].cpu().numpy() > 0.5
            gt_c = gt[c].cpu().numpy() > 0.5

            if np.sum(gt_c) == 0 and np.sum(pred_c) == 0:
                dice = 1.0
                assd_val = 0.0
                hd_val = 0.0
            elif np.sum(gt_c) == 0 or np.sum(pred_c) == 0:
                dice = 0.0
                assd_val = np.nan
                hd_val = np.nan
            else:
                intersection = np.logical_and(pred_c, gt_c).sum()
                dice = 2 * intersection / (pred_c.sum() + gt_c.sum())
                try:
                    assd_val = assd(pred_c, gt_c)
                    hd_val = hd95(pred_c, gt_c)
                except:
                    assd_val = np.nan
                    hd_val = np.nan

            dice_scores[c].append(dice)
            assd_scores[c].append(assd_val)
            hd_scores[c].append(hd_val)


    gg = {}
    for i, name in enumerate(class_names):
        if np.nanmean(assd_scores[i]) < 0.1 or np.nanmean(dice_scores[i]) > 0.99:
            continue
        else:
            gg[i] = name

    results = []
    for i, name in gg.items():
        dice_mean = np.nanmean(dice_scores[i])
        assd_mean = np.nanmean(assd_scores[i])
        hd_mean = np.nanmean(hd_scores[i])
        results.append({
            'Class': names_to_abb[name],
            'Dice': round(dice_mean, 4),
            'ASSD': round(assd_mean, 2),
            'HD': round(hd_mean, 2)
        })

    mean_dice = np.nanmean([np.nanmean(dice_scores[i]) for i in gg.keys()])
    mean_assd = np.nanmean([np.nanmean(assd_scores[i]) for i in gg.keys()])
    mean_hd = np.nanmean([np.nanmean(hd_scores[i]) for i in gg.keys()])
    results.append({
        'Class': 'Average',
        'Dice': round(mean_dice, 4),
        'ASSD': round(mean_assd, 4),
        'HD': round(mean_hd, 4)
    })

    return results

def evaluate_multiclass_segmentation2(preds, gts, class_names):
    num_classes = len(class_names)
    dice_scores = [[] for _ in range(num_classes)]
    assd_scores = [[] for _ in range(num_classes)]
    hd_scores = [[] for _ in range(num_classes)]

    for pred, gt in zip(preds, gts):
        for c in range(num_classes):
            pred_c = pred[c].cpu().numpy() > 0.5
            gt_c = gt[c].cpu().numpy() > 0.5

            if np.sum(gt_c) == 0 and np.sum(pred_c) == 0:
                dice = 1.0
                assd_val = 0.0
                hd_val = 0.0
            elif np.sum(gt_c) == 0 or np.sum(pred_c) == 0:
                dice = 0.0
                assd_val = np.nan
                hd_val = np.nan
            else:
                intersection = np.logical_and(pred_c, gt_c).sum()
                dice = 2 * intersection / (pred_c.sum() + gt_c.sum())
                try:
                    assd_val = assd(pred_c, gt_c)
                    hd_val = hd95(pred_c, gt_c)
                except:
                    assd_val = np.nan
                    hd_val = np.nan

            dice_scores[c].append(dice)
            assd_scores[c].append(assd_val)
            hd_scores[c].append(hd_val)

    results = []
    for i, name in enumerate(class_names):
        dice_mean = np.nanmean(dice_scores[i])
        assd_mean = np.nanmean(assd_scores[i])
        hd_mean = np.nanmean(hd_scores[i])
        results.append({
            'Class': name,
            'Dice': round(dice_mean, 4),
            'ASSD': round(assd_mean, 2),
            'HD': round(hd_mean, 2)
        })

    # 每张图的平均 Dice（宏平均）
    mean_dice = np.nanmean([np.mean(s) for s in dice_scores])
    results.append({
        'Class': 'Average',
        'Dice': round(mean_dice, 4),
        'ASSD': '-',
        'HD': '-'
    })

    return results

def save_visualization(image, gt_mask, pred_mask, class_names, save_path, alpha=0.5):
    """
    image: (3, H, W), gt_mask/pred_mask: (C, H, W)
    """
    os.makedirs(save_path, exist_ok=True)
    image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    gt_overlay = image_np.copy().astype(np.float32)
    pred_overlay = image_np.copy().astype(np.float32)
    cmap = cm.get_cmap('tab20', len(class_names))

    for i in range(len(class_names)):
        gt = gt_mask[i].cpu().numpy() > 0.5
        pred = pred_mask[i].cpu().numpy() > 0.5

        color = np.array(cmap(i)[:3]) * 255  # RGB
        gt_overlay[gt] = alpha * color + (1 - alpha) * gt_overlay[gt]
        pred_overlay[pred] = alpha * color + (1 - alpha) * pred_overlay[pred]

    gt_overlay = np.clip(gt_overlay, 0, 255).astype(np.uint8)
    pred_overlay = np.clip(pred_overlay, 0, 255).astype(np.uint8)

    vis_image = np.concatenate([image_np, gt_overlay, pred_overlay], axis=1)

    save_file = os.path.join(save_path, 'vis.png')
    cv2.imwrite(save_file, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

def save_visualization_per_class(image, gt_mask, pred_mask, class_names, save_path, alpha=0.5):
    """
    image: (3, H, W), gt_mask/pred_mask: (C, H, W)
    """
    os.makedirs(save_path, exist_ok=True)
    image_np = (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cmap = cm.get_cmap('tab20', len(class_names))

    for i, name in enumerate(class_names):
        gt = gt_mask[i].cpu().numpy() > 0.5
        pred = pred_mask[i].cpu().numpy() > 0.5

        # 融合显示
        gt_overlay = image_np.copy()
        pred_overlay = image_np.copy()

        gt_color = np.array(cmap(i)[:3]) * 255
        pred_color = np.array(cmap(i)[:3]) * 255

        gt_overlay[gt] = alpha * gt_color + (1 - alpha) * gt_overlay[gt]
        pred_overlay[pred] = alpha * pred_color + (1 - alpha) * pred_overlay[pred]

        concat = np.concatenate([image_np, gt_overlay.astype(np.uint8), pred_overlay.astype(np.uint8)], axis=1)
        save_file = os.path.join(save_path, f'{name}.png')
        cv2.imwrite(save_file, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))

def evaluate_model(model, image_dir, json_path, class_names, device=device):
    # load model

    model.eval()
    model.to(device)

    # build dataset
    dataset = MyDataset(image_dir, json_path, class_names=class_names)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    preds_all = []
    gts_all = []
    idx=0
    with torch.no_grad():
        for image, mask in tqdm(dataloader):
            image = image.to(device)  # [1, 3, H, W]
            mask = mask.to(device)    # [1, C, H, W]

            output, _ = model(image, mode='eval_topk')     # 假设模型输出是 [1, C, H, W]
            output = torch.sigmoid(output)

            preds_all.append(output[0].cpu())
            gts_all.append(mask[0].cpu())
            idx = idx + 1
            # 保存可视化
            vis_path = os.path.join('./visual', f'sample_{idx:03d}')
            save_visualization(image[0], mask[0], output[0], class_names, vis_path)

    # evaluate
    results = evaluate_multiclass_segmentation(preds_all, gts_all, class_names)

    # print
    df = pd.DataFrame(results)
    df.to_csv('evaluation_results.csv', index=False)

if __name__ == '__main__':
    # 示例配置
    model_path = 'ckpt/20250728134819_setr/ours_dataset_cur.pth'
    json_path = '../data/lunao_seg/annotations.json'
    val_image_dir = '../data/lunao_seg/val'

    model = BACMoE_Model(num_classes=19)
    model.load_state_dict(torch.load(model_path, map_location=device))

    class_names = {'丘脑': 0, '中脑': 1, '中脑导水管': 2, '侧脑室前角': 3, '侧脑室后角': 4, '大脑外侧裂': 5,
                   '大脑实质': 6, '大脑镰': 7, '小脑半球': 8, '小脑蚓部': 9, '穹窿柱': 10, '第三脑室': 11, '胼胝体': 12,
                   '脉络丛': 13, '脑岛': 14, '透明隔': 15, '透明隔腔': 16, '颅后窝池': 17, '颅骨光环': 18}
    # class_names = {'三尖瓣开放': 0, '二尖瓣开放': 1, '切面大框': 2, '右室壁': 3, '右心室': 4, '右心房': 5, '右肺': 6,
    #                '室间隔': 7, '左室壁': 8, '左心室': 9, '左心房': 10, '左肺': 11, '心脏面积': 12, '房间隔': 13,
    #                '肋骨': 14, '肺静脉角': 15, '胸腔面积': 16, '脊柱': 17, '降主动脉': 18}

    evaluate_model(model, val_image_dir, json_path, class_names.keys())

