import os
import torch
from sklearn.metrics import accuracy_score
import numpy as np

def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
        torch.save(state, best_path)

def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
    epoch = checkpoint.get('epoch', None)
    best_acc = checkpoint.get('best_acc', None)
    return model, optimizer, epoch, best_acc

def top1_accuracy(output, target):
    preds = output.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    return accuracy_score(target, preds)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
