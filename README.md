# YOLOv4-Tiny-Satellite-Object-Detection

This project demonstrates how to detect swimming pools and cars using satellite imagery with the YOLOv4-tiny model. The original dataset was sourced from Kaggle and converted from PASCAL VOC XML format to YOLOv4 format.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Results](#results)
- [References](#references)

## Overview
This repository contains the code and instructions to train a YOLOv4-tiny object detection model to identify swimming pools and cars in satellite images. YOLOv4-tiny is a smaller, faster version of YOLOv4, suitable for deployment on devices with limited computational power.

## Dataset
The dataset used for this project consists of satellite images labeled for swimming pools and cars. The dataset was originally in PASCAL VOC XML format and was converted to YOLOv4 format. The dataset can be downloaded from the following Kaggle links:
- [Swimming Pool and Car Detection](https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection)
- [Car and Swimming Pool Satellite Imagery](https://www.kaggle.com/datasets/tekbahadurkshetri/car-and-swimming-pool-satellite-imagery)

## Installation
To set up the environment and get started with training the YOLOv4-tiny model, follow these steps:

1. Clone the darknet repository:
    ```bash
    git clone https://github.com/AlexeyAB/darknet
    ```

2. Create a directory in Google Drive for the YOLOv4-tiny files and datasets:
    ```bash
    mkdir -p /mydrive/yolov4-tiny/training
    ```

3. Upload the necessary files (`archive.zip`, `yolov4-tiny-custom.cfg`, `obj.data`, `obj.names`, `process.py`) to the `/mydrive/yolov4-tiny` directory on your Google Drive.

## Training the Model
1. Mount Google Drive in Colab and link the folder:
    ```python
    from google.colab import drive
    drive.mount('/content/gdrive')
    !ln -s /content/gdrive/My\ Drive/ /mydrive
    ```

2. Enable GPU and OpenCV in the darknet Makefile:
    ```bash
    sed -i 's/OPENCV=0/OPENCV=1/' Makefile
    sed -i 's/GPU=0/GPU=1/' Makefile
    sed -i 's/CUDNN=0/CUDNN=1/' Makefile
    sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
    sed -i 's/LIBSO=0/LIBSO=1/' Makefile
    ```

3. Build darknet:
    ```bash
    make
    ```

4. Copy files from Google Drive to the darknet directory and unzip the dataset:
    ```bash
    cp /mydrive/yolov4-tiny/archive.zip ../
    unzip ../archive.zip -d data/
    cp /mydrive/yolov4-tiny/yolov4-tiny-custom.cfg ./cfg
    cp /mydrive/yolov4-tiny/obj.names ./data
    cp /mydrive/yolov4-tiny/obj.data  ./data
    cp /mydrive/yolov4-tiny/process.py ./
    ```

5. Create `train.txt` and `test.txt` files:
    ```bash
    python process.py
    ```

6. Download the pre-trained weights:
    ```bash
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
    ```

7. Train the YOLOv4-tiny model:
    ```bash
    ./darknet detector train data/obj.data cfg/yolov4-tiny-custom.cfg yolov4-tiny.conv.29 -dont_show -map
    ```

## Testing the Model
1. To test the model on a single image:
    ```bash
    ./darknet detector test data/obj.data cfg/yolov4-tiny-custom.cfg /mydrive/yolov4-tiny/training/yolov4-tiny-custom_best.weights /path/to/image.jpg -thresh 0.5
    ```

2. Display the prediction:
    ```python
    import cv2
    import matplotlib.pyplot as plt

    def imShow(path):
        image = cv2.imread(path)
        height, width = image.shape[:2]
        resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)
        fig = plt.gcf()
        fig.set_size_inches(18, 10)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))

    imShow('predictions.jpg')
    ```

## Results
Include any relevant images, charts, or metrics that demonstrate the performance of your trained model.

## References
- [Darknet GitHub Repository](https://github.com/AlexeyAB/darknet)
- [Swimming Pool and Car Detection Dataset](https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection)
- [Car and Swimming Pool Satellite Imagery](https://www.kaggle.com/datasets/tekbahadurkshetri/car-and-swimming-pool-satellite-imagery)
- [YOLOv4-tiny Paper](https://arxiv.org/abs/2004.10934)
- [Colab Notebook for Training YOLOv4](https://colab.research.google.com/drive/1S6xZATUmcS3eHZtFmGBnu69v90L-gg8A?usp=sharing)

## process.py
```python
import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)
current_dir = 'data/data/train'

# Percentage of images to be used for the test set
percentage_test = 10

# Create and/or truncate train.txt and test.txt
file_train = open('data/train.txt', 'w')
file_test = open('data/test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)

for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if counter == index_test:
        counter = 1
        file_test.write(current_dir + "/" + title + '.jpg' + "\n")
    else:
        file_train.write(current_dir + "/" + title + '.jpg' + "\n")
        counter = counter + 1
