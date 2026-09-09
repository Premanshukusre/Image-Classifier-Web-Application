# 🍎 Advanced AI Image Classifier Web Application

An end-to-end Deep Learning & Botanical Diagnostic Web Application for classifying **Apple leaf diseases** using Convolutional Neural Networks (CNN) and Transfer Learning.

> 🌟 **Project Status:** Complete & Deployed

> Features a state-of-the-art Light Theme AI Diagnostic Dashboard, Live WebCam Scanner, Tabbed Botanical Intelligence, Probability Bar Charts, Visual Leaf Inspector, PDF Export, and Local Session Scan History.

---

## 📌 Project Overview

Plant diseases can significantly affect crop quality and agricultural productivity. Early identification of diseases from leaf images can help farmers and agricultural professionals take appropriate action.

This project combines deep learning model training with an **Advanced AI Diagnostic Dashboard**:

- **Deep Learning Model**: High-accuracy CNN trained on Apple leaf disease dataset.
- **Modern Light UI**: Glassmorphic responsive interface with light background.
- **Drag-and-Drop & WebCam Scanner**: Instant image selection or live photo capture.
- **1-Click Test Presets**: Built-in sample gallery for instant demonstration.
- **Comprehensive Probability Breakdown**: Visual probability bars across all 4 leaf categories.
- **Botanical Intelligence Base**: Structured symptoms, organic remedies, chemical treatments, and prevention guidelines.
- **Visual Leaf Inspector**: HTML5 Canvas filters (Normal, High Contrast, Edge Highlight, Thermal Map).
- **Session Scan History**: Auto-saves past scans locally with 1-click reload.
- **Export Diagnostic Reports**: Print and PDF report generation for orchard management.

---

## 🎯 Objective

Classify apple leaf images into one of four categories:
1. **Apple Scab** (`Venturia inaequalis`)
2. **Black Rot** (`Botryosphaeria obtusa`)
3. **Cedar Apple Rust** (`Gymnosporangium juniperi-virginianae`)
4. **Healthy Apple Leaf**

---

## 🌱 Classes & Disease Intel

| Class | Display Name | Severity Level | Pathogen |
|---|---|---|---|
| `Apple___Apple_scab` | Apple Scab | Moderate to High | *Venturia inaequalis* |
| `Apple___Black_rot` | Black Rot | High Severity Risk | *Botryosphaeria obtusa* |
| `Apple___Cedar_apple_rust` | Cedar Apple Rust | Moderate Pathogen | *Gymnosporangium juniperi-virginianae* |
| `Apple___healthy` | Healthy Leaf | Optimal Health | None detected |

---

## 📊 Dataset Distribution

| Dataset Split | Images |
|---|---:|
| Training | 2,281 |
| Validation | 630 |
| Testing | 254 |
| **Total** | **3,165** |

---

## 🛠️ Installation & Quick Start

### 1. Prerequisites
- Python 3.10+
- Flask 3.x
- TensorFlow 2.x

### 2. Environment Setup
```bash
# Clone Repository
git clone https://github.com/Premanshukusre/Image-Classifier-Web-Application.git
cd Image-Classifier-Web-Application

# Activate Virtual Environment & Install Requirements
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000`.

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
```

---

## 💻 Tech Stack
- **Backend**: Python, Flask, TensorFlow / Keras, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Custom Glassmorphism Design System), JavaScript (ES6 AJAX, Canvas API, WebCam API)
- **Model**: Custom CNN / Transfer Learning (`models/apple_disease_model.keras`)
