import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw
from ultralytics import YOLO
import os

# ----------- Classification -----------
def classify_image(img, st):
    model_path = os.path.join('.', 'runs', 'classify', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
    results = model.predict(img)
    result = results[0]
    class_names = result.names
    probs = result.probs.data.tolist()
    class_name = class_names[np.argmax(probs)].upper()

    width = img.shape[0]
    cv.putText(img, class_name, (width - 80, 60), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)

    st.subheader('Classification Result')
    st.image(img, channels="BGR", use_column_width=True)

# ----------- Object Detection -----------
def detect_image(img, confidence, st):
    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
    model = YOLO(model_path)
    results = model.predict(img, conf=confidence)
    result = results[0]
    im_array = result.plot()
    im = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB

    st.subheader('Detection Result')
    st.image(im, channels="RGB", use_column_width=True)

# ----------- Segmentation -----------
def segment_image(img, st):
    model_path = os.path.join('.', 'runs', 'segment', 'train', 'weights', 'best.pt')
    H, W, _ = img.shape
    model = YOLO(model_path)
    results = model.predict(img)
    st.write(f"[INFO] Number of masks detected: {len(results[0].masks)}")

    mask_out = None
    for result in results:
        for _, mask_ in enumerate(result.masks.data):
            mask_gray = mask_.numpy() * 255
            mask_gray = cv.resize(mask_gray, (W, H))
            mask_out = mask_gray if mask_out is None else cv.bitwise_or(mask_out, mask_gray)

    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    for result in results:
        for mask_ in result.masks:
            polygon = mask_.xy[0]
            draw.polygon(polygon, outline=(0, 255, 0), width=5)

    st.subheader('Segmentation Result')
    cols = st.columns(2)
    cols[0].image(mask_out, clamp=True, channels='GRAY', use_column_width=True)
    cols[1].image(im, channels='RGB', use_column_width=True)

# ----------- Main Streamlit App -----------
def main():
    st.title("YOLOv8 Image Analysis")
    st.sidebar.title("Settings")
    
    task = st.sidebar.selectbox("Choose Task", ("Classification", "Detection", "Segmentation"))
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        if task == "Classification":
            classify_image(img_bgr, st)
        elif task == "Detection":
            confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
            detect_image(img_bgr, confidence, st)
        elif task == "Segmentation":
            segment_image(img_bgr, st)

if __name__ == "__main__":
    main()
