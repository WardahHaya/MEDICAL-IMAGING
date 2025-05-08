
# 🧠 YOLOv8 for Medical Imaging
YOLO (You Only Look Once) is renowned for its ability to detect objects in a single pass through an image, making it one of the most efficient and accurate object detection algorithms available. 🎯

The latest release — YOLOv8, introduced by Ultralytics in January 2023 — brings significant improvements in speed, accuracy, and versatility across various computer vision tasks.

In this project, I focus on three core tasks enabled by YOLOv8:

🔍 Image Classification

🧭 Object Detection

🩻 Instance Segmentation

The primary goal is to explore how YOLOv8 can be effectively applied in the field of medical imaging to detect, classify, and segment anomalies and diseases. 🧪💊.


## ⚙️ What's New in YOLOv8?
YOLOv8 introduces several architectural and functional enhancements that set it apart from its predecessors:

🧠 New Backbone Network
YOLOv8 utilizes the powerful Darknet-53 as its backbone, improving the model’s ability to extract rich and deep features from input images.

🎯 Anchor-Free Detection
Departing from traditional anchor-based approaches, YOLOv8 implements an anchor-free detection head, allowing the model to predict object centers directly. This simplifies training and improves detection accuracy, especially for small or irregularly shaped objects.

📉 Updated Loss Function
A newly designed loss function has been introduced to better optimize bounding box regression and classification accuracy, leading to more precise predictions.

These improvements make YOLOv8 not only faster and more lightweight but also more robust and adaptable for real-world applications, especially in sensitive domains like medical diagnostics.

## 🔬 Project Overview
In this project, I focus on three major computer vision tasks using YOLOv8, all accessible through the Streamlit web application:

1. **Classification:** Utilize the YOLOv8 model to classify medical images into three categories: COVID-19, Viral Pneumonia, and Normal, using the [COVID-19 Image 
Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

2. **Object Detection:** Employ YOLOv8 for detecting Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets in blood cell images using the [RBC and WBC Blood Cells Detection 
Dataset](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs).

3. **Segmentation:** Use YOLOv8 for segmenting breast ultrasound images with the [Breast Ultrasound Images Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset).

## 🖥️ Streamlit Interface
To ensure smooth and intuitive interaction with the YOLOv8 model, this project features a clean, user-friendly Streamlit web interface. Users can explore each task—classification, detection, and segmentation—through a dedicated tab.

Below are screenshots showcasing different parts of the application:

📘 About Page
Provides an overview of the project, background on YOLOv8, and instructions for use.
![About](https://github.com/WardahHaya/MEDICAL-IMAGING/blob/main/intro_ss.png)

🧭 Object Detection
Detects Red Blood Cells, White Blood Cells, and Platelets in blood smear images.
![Object]https://github.com/WardahHaya/MEDICAL-IMAGING/blob/main/detection_ss.png

🩺 Classification
Classifies chest X-ray images as COVID-19, Viral Pneumonia, or Normal.
![Classification](https://github.com/WardahHaya/MEDICAL-IMAGING/blob/main/classification_ss.png)


🩻 Segmentation
Segments tumor regions in breast ultrasound scans for enhanced visualization.
![Segment](https://github.com/sevdaimany/YOLOv8-Medical-Imaging/blob/master/segmentation/segmentation_screenshot.png)


**Run the Streamlit App:**

Start the Streamlit app to see our project in action:
```bash
streamlit run app.py
```



