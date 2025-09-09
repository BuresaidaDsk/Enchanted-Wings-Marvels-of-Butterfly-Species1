import argparse, os, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import build_model
from utils import save_checkpoint, top1_accuracy, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset folder with train/val/test subfolders')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def get_loaders(data_dir, batch_size, num_workers):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_dataset.classes

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for images, labels in tqdm(loader, desc='Train'):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        running_acc += (outputs.argmax(1) == labels).sum().item()
        n += images.size(0)
    return running_loss / n, running_acc / n

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Val'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            running_acc += (outputs.argmax(1) == labels).sum().item()
            n += images.size(0)
    return running_loss / n, running_acc / n

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    train_loader, val_loader, classes = get_loaders(args.data_dir, args.batch_size, args.num_workers)
    num_classes = len(classes)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    model = build_model(num_classes, backbone=args.backbone, pretrained=True, freeze_backbone=args.freeze_backbone)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_acc = 0.0
    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        is_best = val_acc > best_acc
        best_acc = max(best_acc, val_acc)
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_acc': best_acc
        }, is_best, args.output_dir, filename=f'checkpoint_epoch_{epoch}.pth')
        print(f'Epoch {epoch} | Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f} | Time {(time.time()-start):.1f}s')

    print('Training finished. Best val acc: {:.4f}'.format(best_acc))

if __name__ == '__main__':
    main()
