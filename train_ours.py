import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from my_dataset import MyDataset
from BACMoE_Model import BACMoE_Model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)

def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)

def dice_metric(output, target):
    output = output > 0
    dice = ((output * target).sum() * 2+0.1) / (output.sum() + target.sum() + 0.1)
    return dice

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def voe_metric(output, target):
    output = output > 0
    voe = ((output.sum() + target.sum()-(target*output).sum().float()*2)+0.1) / (output.sum() + target.sum()-(target*output).sum().float() + 0.1)
    return voe.item()

def rvd_metric(output, target):
    output = output > 0
    rvd = ((output.sum() / (target.sum() + 0.1) - 1) * 100)
    return rvd.item()

def acc_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    acc = (target==output).sum().float() / target.shape[0]
    return acc

def sen_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    p = (target*output).sum().float()
    sen = (p+0.1) / (output.sum()+0.1)
    return sen

def spe_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    tn = target.shape[0] - (target.sum() + output.sum() - (target*output).sum().float())
    spe = (tn+0.1) / (target.shape[0] - output.sum()+0.1)
    return spe

def router_entropy_loss(gates):
    """
    gates: tensor, shape [B, num_experts]
    """
    p = gates.mean(dim=0)
    entropy = - (p * torch.log(p + 1e-9)).sum()
    return entropy

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score


def train_epoch(epoch, model, dl, optimizer, criterion, criterion2):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, ii = 0, 0, 0

    for x2, mask in bar:
        loss_list = []
        outputs, logits = model(x2.to(device), mode='train')
        mask = mask.to(device)
        B, C, H, W = outputs.shape
        for c in range(C):
            pred_c = outputs[:, c:c+1, :, :].unsqueeze(1)   # (B, 1, H, W)
            mask_c = mask[:, c:c+1, :, :].unsqueeze(1)      # (B, 1, H, W)
            loss_c = criterion(pred_c,mask_c) + 0.6 * criterion2(pred_c,mask_c)
            loss_list.append(loss_c)
        logits_fourier, logits_fuzzy = logits
        loss = torch.stack(loss_list).sum() + 0.4 * (router_entropy_loss(logits_fourier) + router_entropy_loss(logits_fuzzy))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(outputs, mask)
        dice_v += dice
        loss_v += loss.item()
        ii += 1
        bar.set_postfix(loss=loss.item(), dice=dice.item())
    return loss_v / ii, dice_v / ii

@torch.no_grad()
def val_epoch(model, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v,acc_v, sen_v, spe_v, ii = 0, 0, 0,0, 0, 0, 0, 0
    for x2, mask in dl:
        outputs, _ = model(x2.to(device), mode='eval_topk')
        mask = mask.to(device)
        loss_v += criterion(outputs, mask).item()
        dice_v += dice_metric(outputs, mask)
        voe_v += voe_metric(outputs, mask)
        rvd_v += rvd_metric(outputs, mask)
        acc_v += acc_m(outputs, mask)
        sen_v += sen_m(outputs, mask)
        spe_v += spe_m(outputs, mask)

        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii, acc_v / ii, sen_v / ii, spe_v / ii


def train(opt):
    model = BACMoE_Model(num_classes=19)

    if opt.w:
        model.load_state_dict(torch.load(opt.w))

    model = model.to(device)
    model = nn.DataParallel(model)

    json_path = '../data/lunao_seg/annotations.json'
    train_image_dir = '../data/lunao_seg/train'
    val_image_dir = '../data/lunao_seg/val'
    
    train_dataset =MyDataset(image_dir=train_image_dir, json_path=json_path, image_size=(512, 512))
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=6, shuffle=True)
    val_dataset = MyDataset(image_dir=val_image_dir, json_path=json_path, image_size=(512, 512))
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=6, shuffle=False)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    criterion2 = DiceLoss()
    # logs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,patience=10)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice, b_acc, b_sen, b_spe,pre_loss, sur_loss =  0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + opt.name
    mkdirs(save_dir)

    w_dice_best = os.path.join(save_dir, 'ours_dataset_cur.pth')
    fout_log = open(os.path.join(save_dir, 'ours_log.txt'), 'w')
    print(len(train_dataset), len(val_dataset), save_dir)
    for epoch in range(opt.max_epoch):
        if not opt.eval:
            train_loss, train_dice = train_epoch(epoch, model, train_dl, optimizer, criterion, criterion2)
        val_loss, val_dice, voe_v, rvd_v, acc_v, sen_v, spe_v = val_epoch(model, val_dl, criterion)
        if best_dice < val_dice:
            best_dice, best_dice_epoch, b_voe, b_rvd,b_acc, b_sen, b_spe = val_dice, epoch, voe_v, rvd_v, acc_v, sen_v, spe_v
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), w_dice_best)
        
        lr = optimizer.param_groups[0]['lr']
        log = "%02d train_loss:%0.3e, train_dice:%0.5f, val_loss:%0.3e, val_dice:%0.5f, lr:%.3e\n best_dice:%.5f, voe:%.5f, rvd:%.5f, acc:%.5f, sen:%.5f, spe:%.5f(%02d)\n" % (
            epoch, train_loss, train_dice, val_loss, val_dice, lr, best_dice, b_voe, b_rvd, b_acc, b_sen, b_spe, best_dice_epoch)
        print(log)
        fout_log.write(log)
        fout_log.flush()
        scheduler.step(val_loss)
    fout_log.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='setr', help='study name')
    parser.add_argument('--batch_size', type=int, default=12, help='batch size')
    parser.add_argument('--input_size', type=int, default=512, help='input size')
    parser.add_argument('--max_epoch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='the dir path to save model weight')
    parser.add_argument('--w', type=str, help='the path of model wight to test or reload')
    parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], help='suffix', default='.png')
    parser.add_argument('--eval', action="store_true", help='eval only need weight')
    parser.add_argument('--test_root', type=str, help='root_dir')

    opt = parser.parse_args()
    train(opt)
