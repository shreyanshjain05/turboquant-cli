#!/bin/bash
# download_phi3.sh — Download Phi-3-mini-4k-instruct weights for TurboQuant
#   chmod +x download_phi3.sh && ./download_phi3.sh
#

set -e

MODEL_DIR="$HOME/.cache/huggingface/hub/phi3_weights"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo "  Downloading Phi-3-mini-4k-instruct"
echo "  Model size: ~7.6 GB (2 shards)"
echo "  Destination: $MODEL_DIR"
echo "============================================"
echo ""

# Shard 1 (~4.7 GB)
echo "[1/2] Downloading model-00001-of-00002.safetensors (~4.7 GB)..."
curl -L -C - --retry 10 --retry-delay 5 \
  -o "$MODEL_DIR/model-00001-of-00002.safetensors" \
  "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/model-00001-of-00002.safetensors"
echo "  Shard 1 done!"
echo ""

# Shard 2 (~2.9 GB)
echo "[2/2] Downloading model-00002-of-00002.safetensors (~2.9 GB)..."
curl -L -C - --retry 10 --retry-delay 5 \
  -o "$MODEL_DIR/model-00002-of-00002.safetensors" \
  "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/model-00002-of-00002.safetensors"
echo "  Shard 2 done!"
echo ""

# Now register into HF cache properly
echo "Registering model into HuggingFace cache..."
source .venv/bin/activate 2>/dev/null || true
python -c "
from huggingface_hub import snapshot_download
import os, shutil

# This will download all the small files (config, tokenizer, etc.)
# and use the cache. The big shards we already have.
path = snapshot_download(
    'microsoft/Phi-3-mini-4k-instruct',
    ignore_patterns=['*.safetensors'],  # skip big files, we have them
)
print(f'Snapshot at: {path}')

# Copy our downloaded shards into the snapshot
src_dir = os.path.expanduser('~/.cache/huggingface/hub/phi3_weights')
for fname in ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']:
    src = os.path.join(src_dir, fname)
    dst = os.path.join(path, fname)
    if os.path.exists(src) and not os.path.exists(dst):
        print(f'Linking {fname} into cache...')
        os.symlink(src, dst)

print('Done! Model is ready.')
"

echo ""
echo "============================================"
echo "  Download complete! You can now run:"
echo ""
echo "  python benchmark.py --real --backend huggingface \\"
echo "      --model microsoft/Phi-3-mini-4k-instruct"
echo "============================================"
