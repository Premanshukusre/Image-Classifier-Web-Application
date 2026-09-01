# 🍎 Image Classifier Web Application

An end-to-end Deep Learning project for classifying **Apple leaf diseases** using Convolutional Neural Networks (CNN) and Transfer Learning, with a web application for image-based prediction.

> 🚧 **Project Status:** In Development  
> Current progress: Dataset preparation, verification, Git/GitHub setup, Python environment, and data loading pipeline completed.

---

## 📌 Project Overview

Plant diseases can significantly affect crop quality and agricultural productivity. Early identification of diseases from leaf images can help farmers and agricultural professionals take appropriate action.

This project aims to develop an image classification system that can identify different conditions of **Apple leaves** from an uploaded image.

The final system will combine:

- Dataset preparation
- Dataset verification
- Duplicate detection
- Data loading and preprocessing
- CNN model development
- Model training
- Model evaluation
- Transfer Learning
- Image preprocessing
- Model evaluation
- Flask web application
- Git/GitHub version control
- Model deployment

---

## 🎯 Objective

The main objective of this project is to build an end-to-end image classification web application capable of classifying an Apple leaf image into one of four categories:

        1. Apple Scab
        2. Apple Black Rot
        3. Apple Cedar Apple Rust
        4. Healthy Apple Leaf

1. Prepare and verify an apple leaf image dataset.
2. Remove or replace duplicate images between dataset splits.
3. Build a CNN-based image classification model.
4. Train the model on apple leaf images.
5. Evaluate the model using accuracy, precision, recall, F1-score, and confusion matrix.
6. Create an image prediction pipeline.
7. Integrate the trained model into a Flask web application.
8. Provide a simple interface for users to upload apple leaf images and receive predictions.
9. Maintain the complete project using Git and GitHub.


The project will also compare a CNN built from scratch with a Transfer Learning model and select the better-performing model for the final web application.

---

## 🌱 Classes

The dataset contains four Apple leaf categories:

| Class | Description |
|---|---|
| `Apple___Apple_scab` | Apple Scab |
| `Apple___Black_rot` | Apple Black Rot |
| `Apple___Cedar_apple_rust` | Apple Cedar Apple Rust |
| `Apple___healthy` | Healthy Apple Leaf |

---

## 📊 Dataset

The project uses the **PlantVillage dataset**, from which the Apple leaf classes were selected.

The dataset was prepared and verified before model development.

### Final Dataset Distribution

| Dataset | Images |
|---|---:|
| Training | 2,281 |
| Validation | 630 |
| Testing | 254 |
| **Total** | **3,165** |

### Image Properties

- Image format: JPEG
- Original image size: 256 × 256 pixels
- Model input size: 224 × 224 pixels
- Color channels: RGB

---

## 🧹 Dataset Preparation

Several dataset quality checks were performed before training.

### 1. Image Validation

All images were inspected for:

- File validity
- Image dimensions
- Image format
- Corrupted images

No corrupted images were found.

### 2. Duplicate Detection

Exact duplicate images were checked between:

- TRAIN ↔ VAL
- TRAIN ↔ TEST
- VAL ↔ TEST

Duplicate images were removed/replaced where necessary.

### Final Duplicate Check

```text
TRAIN ↔ VAL   : 0
TRAIN ↔ TEST  : 0
VAL ↔ TEST    : 0
