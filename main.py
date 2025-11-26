"""
Simple GUI to run inference with an ONNX YOLO model and display the annotated image.

Usage:
  - Put `best_animal_v3.onnx` in the project folder or update the MODEL_PATH variable.
  - Activate your virtualenv, then run:
      python onnx_infer_gui.py

The script opens a file dialog to pick an image, runs the model, and displays the annotated result.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np


MODEL_PATH = "models/best_animal_v3.onnx"
DEFAULT_IMG_SIZE = 640


def load_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics is not installed. Install with: pip install ultralytics") from e
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    # Explicitly set task to 'detect' to avoid ambiguity for ONNX models
    return YOLO(model_path, task='detect')


class InferenceApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("ONNX Inference GUI")
        self.img_label = tk.Label(root)
        self.img_label.pack(padx=10, pady=10)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=(0,10))

        self.open_btn = tk.Button(btn_frame, text="Open Image", command=self.open_image)
        self.open_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(btn_frame, text="Save Result", state=tk.DISABLED, command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        self.quit_btn = tk.Button(btn_frame, text="Quit", command=root.quit)
        self.quit_btn.pack(side=tk.LEFT, padx=5)

        self.annotated_img = None
        self.photo = None

    def open_image(self):
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")])
        if not path:
            return
        try:
            self.run_inference(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_inference(self, image_path: str):
        # Run model.predict using ultralytics YOLO wrapper (handles ONNX)
        # Set device to auto; you can change device if desired.
        results = self.model.predict(source=image_path, imgsz=DEFAULT_IMG_SIZE, conf=0.25, verbose=False)
        if not results:
            raise RuntimeError("No results returned from model")
        r = results[0]
        # r.plot() returns a numpy image (RGB)
        try:
            annotated = r.plot()  # returns HxWx3 uint8 (RGB)
        except Exception:
            # Fallback: try to access r.orig_img and draw boxes manually (not implemented)
            raise RuntimeError("Model returned results but could not produce a plotted image")

        # Convert to PIL Image (RGB)
        if isinstance(annotated, np.ndarray):
            img = Image.fromarray(annotated)
        else:
            # If already PIL image
            img = annotated

        self.annotated_img = img
        self.show_image(img)
        self.save_btn.config(state=tk.NORMAL)

    def show_image(self, pil_image: Image.Image):
        # Resize image to fit window if too large
        max_w, max_h = 1000, 800
        w, h = pil_image.size
        scale = min(1.0, max_w / w, max_h / h)
        if scale < 1.0:
            display_img = pil_image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        else:
            display_img = pil_image
        self.photo = ImageTk.PhotoImage(display_img)
        self.img_label.config(image=self.photo)

    def save_image(self):
        if self.annotated_img is None:
            return
        path = filedialog.asksaveasfilename(title="Save annotated image", defaultextension=".jpg",
                                            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not path:
            return
        self.annotated_img.save(path)
        messagebox.showinfo("Saved", f"Annotated image saved to {path}")


def main():
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    root = tk.Tk()
    app = InferenceApp(root, model)
    root.mainloop()


if __name__ == '__main__':
    main()
