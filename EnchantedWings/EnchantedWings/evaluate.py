import argparse, os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

from model import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def get_test_loader(data_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, test_dataset.classes

def plot_confusion(cm, classes, out_path='confusion.png'):
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(out_path, bbox_inches='tight')
    print('Saved confusion matrix to', out_path)

def main():
    args = parse_args()
    loader, classes = get_test_loader(args.data_dir, args.batch_size, args.num_workers)
    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model_state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    num_classes = None
    for k,v in model_state.items():
        if k.endswith('.fc.weight') or 'classifier.1.weight' in k:
            num_classes = v.shape[0]
            break
    if num_classes is None:
        raise RuntimeError('Could not infer num_classes from checkpoint.')

    model = build_model(num_classes, backbone=args.backbone, pretrained=False)
    model.load_state_dict(model_state)
    model = model.to(args.device)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(args.device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())
    cm = confusion_matrix(all_targets, all_preds)
    print(classification_report(all_targets, all_preds, target_names=classes))
    plot_confusion(cm, classes)

if __name__ == '__main__':
    main()
