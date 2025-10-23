# Animal YOLO - Ocean Park Animal Detector

YOLOv8-based object detection system for identifying animals using the Ocean Park Animal Detector dataset.

## Project Overview

This project trains a YOLOv8 nano model to detect and classify animals in images. The model is optimized for deployment on resource-constrained devices like Raspberry Pi while maintaining high accuracy.

## Requirements

### Hardware
- **Training:** NVIDIA GPU recommended (tested on RTX 4050 with 6GB VRAM)
- **Inference:** CPU or GPU (ONNX export compatible with Raspberry Pi)

### Software
- Python 3.11+
- CUDA 11.8+ (for GPU training)
- 8GB+ RAM (16GB recommended for training)

## Installation

### Step 1: Clone and Setup Virtual Environment

```powershell
# Navigate to project directory
cd "C:\Users\yueny\OneDrive\Documents\Animal YOLO"

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install PyTorch with CUDA Support

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Verify CUDA Setup

```powershell
python check_cuda.py
```

Expected output:
```
CUDA is available - GPU training enabled!
GPU 0: NVIDIA GeForce RTX 4050
NumPy-PyTorch compatibility: OK
```

## Project Structure

```
Animal YOLO/
├── AI_training.py                          # Training script
├── check_cuda.py                           # CUDA verification utility
├── requirements.txt                        # Python dependencies
├── Ocean Park Animal Detector.v1.../      # Dataset directory
│   ├── data.yaml                          # Dataset configuration
│   ├── train/                             # Training images & labels
│   ├── valid/                             # Validation images & labels
│   └── test/                              # Test images & labels
└── runs/                                   # Training outputs (auto-generated)
    └── detect/
        └── Animal_Detector_Model/         # Model checkpoints & metrics
```

## Training

### Basic Training

```powershell
python AI_training.py
```

### Training Configuration

The training script (`AI_training.py`) is configured with:
- **Model:** YOLOv8 nano (`yolov8n.pt`)
- **Epochs:** 75
- **Image Size:** 640x640
- **Batch Size:** 10
- **Device:** Auto-detect (GPU if available, else CPU)

### Modify Training Parameters

Edit `AI_training.py` to customize:

```python
results = model.train(
    data='Ocean Park Animal Detector.v1-animal-dataset-v1.yolov8/data.yaml',
    epochs=75,          # Number of training epochs
    imgsz=640,          # Image size
    batch=10,           # Batch size (reduce if GPU memory issues)
    device=device,      # 0 for GPU, 'cpu' for CPU
    name='Animal_Detector_Model',
)
```

### GPU Memory Issues?

If you encounter CUDA out-of-memory errors:

```python
batch=4,            # Reduce batch size
imgsz=416,          # Reduce image size
```

## Monitoring Training

Training progress is logged to:
- **Console:** Real-time metrics (loss, precision, recall)
- **TensorBoard:** `runs/detect/Animal_Detector_Model/`

View TensorBoard:
```powershell
tensorboard --logdir=runs/detect
```

## Evaluation

After training completes, the script automatically:
1. Validates the model on the validation set
2. Exports to ONNX format for deployment

Results are saved in:
```
runs/detect/Animal_Detector_Model/
├── weights/
│   ├── best.pt          # Best checkpoint
│   └── last.pt          # Last checkpoint
├── best.onnx            # ONNX export
└── results.png          # Training metrics plot
```

## Inference

### Python Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/Animal_Detector_Model/weights/best.pt')

# Run inference
results = model.predict('path/to/image.jpg', save=True)
```

### ONNX Inference (Raspberry Pi)

The exported ONNX model can be deployed on Raspberry Pi or other edge devices:

```python
import cv2
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession('best.onnx')

# Run inference
# (See Ultralytics ONNX inference documentation)
```

## Expected Performance

### Training Time (RTX 4050)
- **75 epochs:** ~30-60 minutes
- **Per epoch:** ~30-60 seconds

### Training Time (CPU)
- **75 epochs:** ~4-8 hours
- **Per epoch:** ~3-6 minutes

### Model Performance
- **Model Size:** ~6MB (YOLOv8 nano)
- **Inference Speed:** 
  - GPU: ~2-5ms per image
  - CPU: ~50-100ms per image
  - Raspberry Pi 4: ~200-500ms per image (ONNX)

## Utilities

### Check CUDA Setup

```powershell
python check_cuda.py
```

Displays:
- PyTorch version
- CUDA availability
- GPU details (name, memory, compute capability)
- NumPy compatibility
- Dependency versions

## Common Issues

### "Numpy is not available" Error

**Cause:** NumPy 2.x incompatible with PyTorch 2.1.2

**Solution:**
```powershell
pip install "numpy<2"
pip install "opencv-python<4.12"
```

### CUDA Out of Memory

**Solution:** Reduce batch size or image size in `AI_training.py`

### CPU-Only PyTorch Installed

**Check:**
```powershell
python -c "import torch; print(torch.__version__)"
```

If output shows `+cpu`, reinstall with CUDA:
```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Dataset

**Source:** Ocean Park Animal Detector v1 (Roboflow)

**Format:** YOLOv8 (images + labels in YOLO format)

**Classes:** See `data.yaml` for class definitions

**Split:**
- Training set: `train/`
- Validation set: `valid/`
- Test set: `test/`

## Model Updates

To retrain or fine-tune:

1. Update dataset in `Ocean Park Animal Detector.v1.../`
2. Modify `data.yaml` if classes change
3. Run training script: `python AI_training.py`

## Resources

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [ONNX Runtime](https://onnxruntime.ai/)



## Acknowledgments

- Ultralytics for YOLOv8 framework
- Ocean Park for dataset
- Roboflow for dataset hosting and annotation tools
