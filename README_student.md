# Fabric Defect Detection using YOLOv8

This project focuses on detecting five types of fabric defects — `cut`, `hole`, `knot`, `stain`, `thread_error` — using the YOLOv8 object detection model. The project includes a Streamlit web application for running inference on images or videos and a Jupyter notebook that documents training and evaluation.

---

## 1) Project Structure

```
.
├── main.py                     # Streamlit app for image/video inference
├── best.pt                     # Trained YOLOv8 weights
├── yolov8n.pt                  # YOLOv8 nano weights (pre-trained)
├── Fabric_Defect_Detection_Using_YOLOV8.ipynb  # Training and evaluation notebook
├── data.yaml                   # Dataset configuration file
├── requirements.txt            # Python dependencies
├── README.dataset.txt          # Dataset origin and licence information
└── README.roboflow.txt         # Dataset export details
```

If you are including the dataset locally, it should follow the YOLO format as referenced by `data.yaml`:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

The `data.yaml` file expects:

```yaml
train: train/images
val: val/images
test: test/images

names:
  - cut
  - hole
  - knot
  - stain
  - thread_error
```

---

## 2) Environment and Installation

Use Python 3.9 or later.

Install dependencies using:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- ultralytics==8.3.187 (YOLOv8)
- streamlit
- opencv-python, Pillow, numpy, pandas, streamlit-webrtc, av

---

## 3) Running the Streamlit App

1. Ensure the trained weights file `best.pt` is available in the project root.
2. Run the application:

```bash
streamlit run main.py
```

3. In the web interface:
- Upload an image or video.
- Adjust confidence and IoU thresholds if needed.
- View the predictions and per-class counts. You can export the results as a CSV file.

---

## 4) Training (Optional)

To reproduce training, run:

```bash
yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=100 batch=16 device=0
```

To validate the trained model:

```bash
yolo detect val model=best.pt data=data.yaml imgsz=640 conf=0.15
```

---

## 5) Results

From the notebook, the model achieved the following indicative results on the test set at confidence 0.15:

| Class         | Precision | Recall | mAP50 |
|---------------|-----------|--------|-------|
| cut           | 0.613     | 1.000  | 0.965 |
| hole          | 0.612     | 0.574  | 0.663 |
| knot          | 0.463     | 0.247  | 0.307 |
| stain         | 0.496     | 0.360  | 0.446 |
| thread_error  | 0.370     | 0.290  | 0.230 |

---

## 6) Dataset and Licence

- Source: Roboflow Universe – Fabric Defects (5-class)  
  https://universe.roboflow.com/jaswant-oyc4f/fabric-defects-5-class-b2mwz

- Licence: CC BY 4.0 (Attribution required)  
  https://creativecommons.org/licenses/by/4.0/

---

## 7) Notes

- YOLOv8 version used: 8.3.187  
- Image size: 640x640  
- Five defect classes as listed above and in `data.yaml`.

---

## 8) References

- Ultralytics YOLOv8 Documentation — https://docs.ultralytics.com/  
- YOLOv8 GitHub — https://github.com/ultralytics/ultralytics  
- Streamlit Documentation — https://docs.streamlit.io/  
- Roboflow Universe Dataset — https://universe.roboflow.com/jaswant-oyc4f/fabric-defects-5-class-b2mwz  
- Creative Commons BY 4.0 — https://creativecommons.org/licenses/by/4.0/
