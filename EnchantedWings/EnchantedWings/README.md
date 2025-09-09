# Enchanted Wings: Marvels of Butterfly Species

This repository contains code to build a butterfly image classification system using transfer learning (PyTorch + torchvision).
It expects your dataset to follow the standard `ImageFolder` layout:

```
dataset/
  train/
    species_001/
      img1.jpg
      img2.jpg
    species_002/
      ...
  val/
    species_001/
    ...
  test/
    species_001/
    ...
```

## What is included
- `train.py` — training script with transfer learning (ResNet50 by default), checkpointing, scheduler, and mixed precision support (if available).
- `infer.py` — inference script to predict on a single image or folder.
- `model.py` — helper to build and load the model.
- `utils.py` — utilities (metrics, saving/loading).
- `requirements.txt` — Python dependencies.
- `example_config.json` — example config for training.
- `export.sh` — helper to export a checkpoint to a scripted model (TorchScript).
- `evaluate.py` — evaluate model on test set and save confusion matrix.
- `LICENSE` — MIT license.

## Quick start (assuming you have Python and GPU configured)
1. Create a virtualenv and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare your dataset in the `dataset/` layout described above.
3. Train:
   ```
   python train.py --data_dir /path/to/dataset --output_dir outputs --epochs 20 --batch_size 32
   ```
4. Evaluate:
   ```
   python evaluate.py --data_dir /path/to/dataset --checkpoint outputs/checkpoint_best.pth
   ```
5. Infer:
   ```
   python infer.py --checkpoint outputs/checkpoint_best.pth --image some_image.jpg
   ```

## Notes
- The code uses transfer learning (default: ResNet50). You can change the backbone in `train.py`.
- Modify data augmentation in `train.py` to suit your needs.
- This project is intended as a starting point and is fully commented.
