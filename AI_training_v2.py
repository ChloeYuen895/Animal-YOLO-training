from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Check GPU availability
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load pretrained YOLOv8 nano detection model
    model = YOLO('yolov8n.pt')

    # Train on your dataset
    results = model.train(
        data='Ocean Park Animal Detector 2.v1i.yolov8\data.yaml',
        epochs=75,          # Increased to 100 for more training
        imgsz=640,          # Standard image size for YOLOv8
        batch=10,            # Increased batch size for better training
        device=device,
        name='Animal_Detector_Model',
        fliplr = 1.0,        # Always apply horizontal flip
        translate = 0.15,  # Increased translation for more variation
        shear = 10.0,      # Increased shear for more variation
        scale = 0.7  # Scale range (min, max) for zooming out only
    )

    # Evaluate on validation set
    metrics = model.val(task='detect')
    print(metrics)

    # Export to ONNX for Raspberry Pi
    model.export(format='onnx', dynamic=True)