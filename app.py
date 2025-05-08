import streamlit as st
import cv2 as cv
import numpy as np
from PIL import Image
import detection.detect as detect
import classification.classify as classify
import segmentation.segment as segment

# Optional: Use this only when you want to re-train models
def train_models():
    detect.train()
    print("[INFO] Training Detection model done!")
    classify.train()
    print("[INFO] Training Classification model done!")

    segment.prepare_input()
    segment.train()
    print("[INFO] Training Segmentation model done!")

def main():
    st.sidebar.title("Settings")
    st.sidebar.subheader("Parameters")

    # Custom sidebar width
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width:300px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width:300px;
            margin-left:-300px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Select app mode
    app_mode = st.sidebar.selectbox('Choose the App Mode', ['About App', 'Object Detection', 'Object Classification', "Object Segmentation"])

    # --- About Page ---
    if app_mode == 'About App':
        st.header("Introduction to YOLOv8")

        st.markdown("<style> p{margin: 10px auto; text-align: justify; font-size:20px;}</style>", unsafe_allow_html=True)
        st.markdown("<p>🚀 Welcome to our YOLOv8-based medical imaging app! YOLO is known for detecting objects in a single pass, making it highly efficient and accurate. 🎯</p>", unsafe_allow_html=True)
        st.markdown("<p>The latest version, YOLOv8 ( by Ultralytics), brings enhancements in performance through innovations like a new backbone, anchor-free detection, and a new loss function. 🌟</p>", unsafe_allow_html=True)

        st.markdown("""<p>🔍 Key improvements:<br>
                    • New backbone: Darknet-53<br>
                    • Anchor-free object detection<br>
                    • Improved loss function</p>""", unsafe_allow_html=True)

        st.markdown("""<p>🎊 YOLOv8 is versatile—supporting classification, detection, and segmentation out of the box. We apply it in medical imaging to classify and detect anomalies. 🧪💊</p>""", unsafe_allow_html=True)

        st.markdown("<p>Let's dive in and explore what YOLOv8 can do! 💡</p>", unsafe_allow_html=True)

    # --- Object Detection ---
    elif app_mode == "Object Detection":
        st.header("Object Detection with YOLOv8")

        st.sidebar.markdown("----")
        confidence = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.35)
        img_file = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=0)
        DEMO_IMAGE = "DEMO_IMAGES/BloodImage_00000_jpg.rf.5fb00ac1228969a39cee7cd6678ee704.jpg"

        if img_file is not None:
            img_bytes = img_file.read()
            img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
            image = np.array(Image.open(img_file))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        detect.predict(img, confidence, st)

    # --- Classification ---
    elif app_mode == "Object Classification":
        st.header("Classification with YOLOv8")

        st.sidebar.markdown("----")
        img_file = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=1)
        DEMO_IMAGE = "DEMO_IMAGES/094.png"

        if img_file is not None:
            img_bytes = img_file.read()
            img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
            image = np.array(Image.open(img_file))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        classify.predict(img, st)

    # --- Segmentation ---
    elif app_mode == "Object Segmentation":
        st.header("Segmentation with YOLOv8")

        st.sidebar.markdown("----")
        img_file = st.sidebar.file_uploader("Upload an image", type=['jpg','jpeg', 'png'], key=2)
        DEMO_IMAGE = "DEMO_IMAGES/benign (2).png"

        if img_file is not None:
            img_bytes = img_file.read()
            img = cv.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
            image = np.array(Image.open(img_file))
        else:
            img = cv.imread(DEMO_IMAGE)
            image = np.array(Image.open(DEMO_IMAGE))

        st.sidebar.text("Original Image")
        st.sidebar.image(image)

        segment.predict(img, st)

if __name__ == "__main__":
    try:
        # Optional: Uncomment if you need to retrain models
        # train_models()
        main()
    except SystemExit:
        pass

        pass
        

