# 🥔 Potato Chip AI — Quality Monitoring System

An end-to-end AI-powered system for monitoring potato chip quality using deep learning and machine learning.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)

## 📋 Overview

This project implements two AI models for potato chip quality control:

| Model | Task | Input | Output |
|---|---|---|---|
| **CNN** | Defect Detection | Chip image | Defective / Non-defective + confidence |
| **Random Forest** | Process Risk | Fryer parameters | Predicted defect percentage |

Both models are served through a **Streamlit dashboard** with a premium dark-themed UI.

## 🗂️ Project Structure

```
potato-chip-ai/
├── notebooks/
│   ├── 01_cnn_model.ipynb        # CNN training notebook
│   └── 02_rf_model.ipynb         # Random Forest training notebook
├── models/                       # Saved models (created by notebooks)
│   ├── cnn_model.h5              # Trained CNN model
│   └── rf_model.pkl              # Trained Random Forest model
├── .streamlit/
│   └── config.toml               # Streamlit theme configuration
├── app.py                        # Streamlit dashboard
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd potato-chip-ai
pip install -r requirements.txt
```

### 2. Configure Kaggle API (for dataset download)

Make sure you have a Kaggle account and API key:

```bash
# Option A: Place kaggle.json in default location
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Option B: Set environment variables
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

### 3. Train the Models

Run the notebooks in order:

```bash
# Open Jupyter and run each notebook
jupyter notebook notebooks/
```

1. **Run `01_cnn_model.ipynb`** — Downloads dataset, trains CNN, saves to `models/cnn_model.h5`
2. **Run `02_rf_model.ipynb`** — Generates synthetic data, trains RF, saves to `models/rf_model.pkl`

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## 🧠 Models

### CNN — Defect Detection

- **Dataset**: PepsiCo Lab Potato Chips Quality Control (~1,000 images)
- **Architecture**: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Dense(128) → Dropout(0.5) → Dense(1, sigmoid)
- **Training**: 20 epochs, Adam optimizer, binary crossentropy, data augmentation
- **Evaluation**: Accuracy, confusion matrix, classification report

### Random Forest — Process Risk

- **Data**: Synthetic fryer process data (1,000 samples)
- **Features**: fryer_temperature, frying_time, moisture_content, slice_thickness, oil_quality_index
- **Target**: defect_percentage (fryer_temperature is the strongest predictor)
- **Tuning**: GridSearchCV over n_estimators, max_depth, min_samples_split
- **Evaluation**: R² score, MAE, RMSE, feature importance plot

## 📊 Streamlit Dashboard

| Tab | Description |
|---|---|
| 🔍 **Defect Detection** | Upload a chip image → CNN classifies as defective/non-defective with confidence score |
| ⚙️ **Process Risk Predictor** | Adjust fryer parameter sliders → RF predicts defect percentage with risk level |

## 🛠️ Tech Stack

- **Python 3.11**
- **TensorFlow / Keras** — CNN model
- **scikit-learn** — Random Forest with GridSearchCV
- **Streamlit** — Interactive dashboard
- **pandas, numpy** — Data manipulation
- **matplotlib, seaborn** — Visualizations
- **kagglehub** — Dataset download

## 📜 License

This project uses the [PepsiCo Lab Potato Chips Quality Control](https://www.kaggle.com/datasets/concaption/pepsico-lab-potato-quality-control) dataset, which is licensed under CC0: Public Domain.
# potato-chip-ai-quality-monitor
