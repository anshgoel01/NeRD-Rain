import os
import argparse
import random
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils
from data_RGB import get_training_data, get_validation_data
from model import MultiscaleNet as myNet
import losses
from warmup_scheduler import GradualWarmupScheduler
import kornia
from torch.utils.tensorboard import SummaryWriter

# ==========================================================
# SEEDS
# ==========================================================
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1234)

# ==========================================================
# ARGUMENTS
# ==========================================================
parser = argparse.ArgumentParser(description='GT-RAIN Training')

parser.add_argument('--train_dir',
                    default='/content/MFDNet-main/GT-RAIN_train',
                    type=str)

parser.add_argument('--val_dir',
                    default='/content/MFDNet-main/GT-RAIN_train',
                    type=str)

parser.add_argument('--model_save_dir',
                    default='./checkpoints',
                    type=str)

parser.add_argument('--session',
                    default='Multiscale_GT_RAIN',
                    type=str)

parser.add_argument('--patch_size',
                    default=128,   # ✅ safer for GPU
                    type=int)

parser.add_argument('--num_epochs',
                    default=300,
                    type=int)

parser.add_argument('--batch_size',
                    default=1,
                    type=int)

parser.add_argument('--val_epochs',
                    default=1,
                    type=int)

args = parser.parse_args()

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================================
# DIRECTORIES
# ==========================================================
model_dir = os.path.join(args.model_save_dir, 'models', args.session)
utils.mkdir(model_dir)

# ==========================================================
# DATA
# ==========================================================
train_dataset = get_training_data(args.train_dir,
                                  {'patch_size': args.patch_size})

val_dataset = get_validation_data(args.val_dir,
                                  {'patch_size': args.patch_size})

print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))

train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=torch.cuda.is_available())

val_loader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=torch.cuda.is_available())

# ==========================================================
# MODEL
# ==========================================================
model_restoration = myNet().to(device)

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    model_restoration = nn.DataParallel(model_restoration)

# ==========================================================
# OPTIMIZER
# ==========================================================
start_lr = 1e-4
end_lr   = 1e-6

optimizer = optim.Adam(model_restoration.parameters(),
                       lr=start_lr,
                       betas=(0.9, 0.999))

# ==========================================================
# SCHEDULER
# ==========================================================
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    args.num_epochs - warmup_epochs,
    eta_min=end_lr
)

scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1,
    total_epoch=warmup_epochs,
    after_scheduler=scheduler_cosine
)

# ==========================================================
# LOSSES
# ==========================================================
criterion_char = losses.CharbonnierLoss().to(device)
criterion_edge = losses.EdgeLoss().to(device)
criterion_fft  = losses.fftLoss().to(device)
criterion_L1   = nn.L1Loss().to(device)

# ==========================================================
# LOGGING
# ==========================================================
writer = SummaryWriter(model_dir)

best_psnr  = 0
best_epoch = 0
iter_count = 0

# ==========================================================
# TRAINING LOOP
# ==========================================================
for epoch in range(1, args.num_epochs + 1):

    epoch_start_time = time.time()
    epoch_loss = 0

    model_restoration.train()

    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):

        optimizer.zero_grad(set_to_none=True)

        target_ = data[0].to(device)
        input_  = data[1].to(device)

        target = kornia.geometry.transform.build_pyramid(target_, 3)
        restored = model_restoration(input_)

        loss_fft  = sum(criterion_fft(restored[j], target[j]) for j in range(3))
        loss_char = sum(criterion_char(restored[j], target[j]) for j in range(3))
        loss_edge = sum(criterion_edge(restored[j], target[j]) for j in range(3))

        loss_l1 = criterion_L1(restored[3], target[1]) + \
                  criterion_L1(restored[5], target[2])

        loss = loss_char + 0.01 * loss_fft + 0.05 * loss_edge + 0.1 * loss_l1

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        iter_count += 1

        writer.add_scalar('loss/total', loss.item(), iter_count)

    writer.add_scalar('loss/epoch', epoch_loss, epoch)

    # ======================================================
    # VALIDATION
    # ======================================================
    if epoch % args.val_epochs == 0:

        model_restoration.eval()
        psnr_val_rgb = []

        with torch.no_grad():
            for data_val in val_loader:

                target = data_val[0].to(device)
                input_ = data_val[1].to(device)

                restored = model_restoration(input_)
                restored = restored[0]

                psnr = utils.torchPSNR(restored, target)
                psnr_val_rgb.append(psnr)

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        writer.add_scalar('val/psnr', psnr_val_rgb, epoch)

        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(model_dir, "model_best.pth"))

        print(f"[Epoch {epoch}] PSNR: {psnr_val_rgb:.4f} | Best: {best_psnr:.4f}")

    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]

    print("--------------------------------------------------")
    print(f"Epoch {epoch}")
    print(f"Time: {time.time() - epoch_start_time:.2f}s")
    print(f"Loss: {epoch_loss:.4f}")
    print(f"LR: {current_lr:.6f}")
    print("--------------------------------------------------")

    torch.save({
        'epoch': epoch,
        'state_dict': model_restoration.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(model_dir, "model_latest.pth"))

writer.close()
