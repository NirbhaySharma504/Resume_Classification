# Conversational AI Resume Classification

This repository contains my **course project** for building an end-to-end **resume classification system** using transformer-based Natural Language Processing (NLP) models.  
The goal of the project is to automatically classify resume text into professional categories, helping reduce manual screening effort and making the selection process faster and more consistent.

## Project Overview

Resumes are often long, unstructured, and written in many different formats. This project explores how modern transformer models can understand resume content and predict the most likely job domain for each resume.

The notebook includes:
- data collection and preparation
- resume text cleaning and preprocessing
- label encoding for multi-class classification
- training and evaluation of multiple transformer models
- final comparison across models
- inference on sample resumes

## Problem Statement

Manually sorting resumes into job domains is time-consuming and can be inconsistent.  
This project aims to build a machine learning system that can classify resumes into categories such as technical, administrative, healthcare, finance, engineering, and other professional domains.

## Dataset

The project uses a combination of:
- the **Kaggle Resume Dataset**
- **web-scraped resumes** from PostJobFree

The dataset covers **25+ resume categories**, making it a multi-class text classification problem.

## Models Used

Three transformer models were implemented and compared:

- **DistilBERT** — lightweight and efficient
- **ELECTRA-Small** — compact and fast
- **BERT-Base** — larger model with higher capacity

## Methodology

The workflow followed in this project is:

1. **Data Collection**  
   Resume data was gathered from the Kaggle dataset and additional web sources.

2. **Data Cleaning & Preprocessing**  
   Text was normalized and cleaned by removing unnecessary artifacts, stopwords, and noisy patterns.

3. **Label Encoding**  
   Resume categories were converted into numeric labels for model training.

4. **Tokenization**  
   Transformer-specific tokenizers were used to convert text into model-ready input IDs and attention masks.

5. **Model Training**  
   Each transformer model was fine-tuned for multi-class classification.

6. **Evaluation**  
   Performance was measured using:
   - accuracy
   - loss
   - precision
   - recall
   - F1-score
   - confusion matrix

7. **Model Comparison**  
   The trained models were compared using summary tables and plots.

## Key Features

- Multi-class resume classification
- Transformer-based deep learning pipeline
- Training and evaluation for multiple architectures
- Mixed-precision training support
- Early-stopping and regularization-focused training setup
- Detailed comparison of model performance
- Resume prediction on sample inputs

## Project Structure

A typical structure used in the notebook is:

```text
Conversational_AI_ADS_Resume_Project/
├── data/
│   ├── processed_resume_data.csv
│   ├── category_mapping.json
│   ├── detailed_model_metrics/
│   └── saved_models/
│       ├── distilbert_resume_classifier/
│       ├── electra_small_resume_classifier/
│       └── bert_base_resume_classifier/
└── notebook.ipynb
```

## How to Run

### 1. Clone or download the repository
Place the notebook and data files in the expected directory structure.

### 2. Open the notebook
Open the project notebook in **Google Colab** or Jupyter Notebook.

### 3. Mount Google Drive
If you are using Colab, mount Google Drive so the notebook can access the dataset and save trained models.

### 4. Install dependencies
Make sure the following packages are available:
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `tqdm`

### 5. Run the notebook cells in order
The notebook is organized to:
- load and preprocess data
- train the models
- save the models and metrics
- compare performance
- run inference on sample resumes

## Inference

After training, the project can predict the category of a new resume text.  
The notebook includes a helper function that tokenizes a resume and returns the predicted professional category.

## Results

The notebook generates:
- validation and test metrics
- classification reports
- confusion matrices
- side-by-side model comparison charts
- category-wise F1-score heatmaps

The final comparison section helps identify:
- the best overall model by accuracy
- the most balanced model by macro F1-score
- category-level strengths and weaknesses

## Future Improvements

Possible extensions for this project include:
- adding more resume categories
- improving preprocessing for noisy web-scraped resumes
- experimenting with larger transformer architectures
- deploying the classifier as a web app or API
- adding explainability features for predictions

## Conclusion

This project demonstrates how transformer models can be used for intelligent resume classification.  
It shows that NLP models can learn from real-world resume text and support automated screening in a practical and scalable way.

---

**Author:** Nirbhay Sharma  
**Project Type:** Course Project
