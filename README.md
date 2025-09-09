# Enchanted-Wings-Marvels-of-Butterfly-Species1
Enchanted Wings: Marvels of Butterfly Species
Project Description
Enchanted Wings: Marvels of Butterfly Species is a deep learning project focused on building a robust butterfly image classification system using transfer learning.
It leverages a dataset of 6,499 butterfly images spanning 75 species, split into training, validation, and test sets. By applying transfer learning with pre-trained CNNs,
the project enhances classification accuracy while reducing training time and computational resources.

Real-world Scenarios
Biodiversity Monitoring: Helps researchers and conservationists identify butterfly species in the field for ecosystem monitoring.

Ecological Research: Supports studies on butterfly behavior, distribution, and environmental changes.

Citizen Science & Education: Provides interactive tools for enthusiasts and students to identify butterfly species and learn about their ecology.

Tech Stack
Programming Language: Python 3.8+

Deep Learning Framework: PyTorch, Torchvision

Data Handling: Pandas, NumPy

Image Processing: Pillow (PIL)

Evaluation & Metrics: scikit-learn (confusion matrix, classification report)

Visualization: Matplotlib

Training Utilities: tqdm (progress bars)

Export/Deployment: TorchScript for model serving

What is included
train.py — training script with transfer learning (ResNet50 by default), checkpointing, scheduler, and mixed precision support (if available).

infer.py — inference script to predict on a single image or folder.

model.py — helper to build and load the model.

utils.py — utilities (metrics, saving/loading).

requirements.txt — Python dependencies.

example_config.json — example config for training.

export.sh — helper to export a checkpoint to a scripted model (TorchScript).

evaluate.py — evaluate model on test set and save confusion matrix.

LICENSE — MIT license.
