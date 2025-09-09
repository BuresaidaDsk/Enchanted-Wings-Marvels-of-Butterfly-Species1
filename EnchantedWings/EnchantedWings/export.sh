#!/bin/bash
# Usage: ./export.sh path/to/checkpoint.pth path/to/output_scripted.pt
CKPT="$1"
OUT="$2"
if [ -z "$CKPT" ] || [ -z "$OUT" ]; then
  echo "Usage: ./export.sh checkpoint.pth output.pt"
  exit 1
fi
python - <<'PY'
import torch
from model import build_model
ckpt = torch.load("$CKPT", map_location='cpu')
state = ckpt.get('model_state', ckpt)
# infer num_classes
num_classes = None
for k,v in state.items():
    if k.endswith('.fc.weight') or 'classifier.1.weight' in k:
        num_classes = v.shape[0]
        break
if num_classes is None:
    raise RuntimeError('Could not infer num_classes')
model = build_model(num_classes, pretrained=False)
model.load_state_dict(state)
model.eval()
scripted = torch.jit.script(model)
scripted.save("$OUT")
print("Saved scripted model to", "$OUT")
PY
