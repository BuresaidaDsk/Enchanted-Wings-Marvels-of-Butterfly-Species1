import argparse, os
import torch
from torchvision import transforms
from PIL import Image
from model import build_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True, help='Image file or folder')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def load_checkpoint_meta(checkpoint_path):
    import torch
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # Try to infer number of classes
    state = ckpt.get('model_state', ckpt)
    # fallback: user must ensure correct num_classes
    return ckpt

def prepare_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

def infer_single(model, img_path, transform, device, classes=None):
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        prob = torch.nn.functional.softmax(out, dim=1)
        top1 = prob.argmax(1).item()
        confidence = prob[0, top1].item()
    label = classes[top1] if classes else str(top1)
    return label, confidence

def main():
    args = parse_args()
    ckpt = load_checkpoint_meta(args.checkpoint)
    # Best practice: user should pass model config. We'll try to infer.
    # If you used the training script, the checkpoint contains the model_state only.
    # For safety, assume num_classes equals size of final layer weights if present.
    model_state = ckpt['model_state'] if 'model_state' in ckpt else ckpt
    # Infer number of classes:
    import torch
    # Try several common keys
    num_classes = None
    for k,v in model_state.items():
        if k.endswith('.fc.weight') or 'classifier.1.weight' in k:
            num_classes = v.shape[0]
            break
    if num_classes is None:
        raise RuntimeError('Could not infer num_classes from checkpoint. Please modify the script with correct num_classes.')

    model = build_model(num_classes, backbone=args.backbone, pretrained=False, freeze_backbone=False)
    model.load_state_dict(model_state)
    model = model.to(args.device)

    transform = prepare_transform()
    if os.path.isdir(args.image):
        files = [os.path.join(args.image, f) for f in os.listdir(args.image) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for f in files:
            label, conf = infer_single(model, f, transform, args.device)
            print(f'{f} -> {label} ({conf:.3f})')
    else:
        label, conf = infer_single(model, args.image, transform, args.device)
        print(f'{args.image} -> {label} ({conf:.3f})')

if __name__ == '__main__':
    main()
