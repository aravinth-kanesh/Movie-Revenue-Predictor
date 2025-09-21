# 🎬 ShowbizPredictor: Movie Revenue Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg?style=for-the-badge)](https://streamlit.io)
[![ML](https://img.shields.io/badge/ML-Random%20Forest%20%7C%20LightGBM-green.svg?style=for-the-badge)]()

A production-ready machine learning application that predicts movie box office revenue using advanced feature engineering and model explainability techniques.

## 🎯 Project Overview

This project demonstrates a complete ML pipeline solving a real business problem: predicting movie revenue to inform investment decisions. Built with modern MLOps practices, it showcases data science skills through an interactive web application.

### Key Achievements:

- Built end-to-end ML pipeline processing 5,000+ movie records (sample from 930k dataset)
- Achieved R² scores of 0.85+ using ensemble methods
- Implemented SHAP explainability for transparent predictions
- Deployed interactive dashboard with 4 distinct analysis modes

## 🔧 Technical Implementation

### Machine Learning Pipeline

- **Data Processing**: Automated preprocessing with missing value handling
- **Feature Engineering**: TF-IDF vectorisation, SVD dimensionality reduction, multi-hot encoding
- **Model Training**: Random Forest (200 trees) and LightGBM (300 estimators) with cross-validation
- **Model Selection**: Systematic comparison using MSE, RMSE, and R² metrics

### Web Application

- **Interactive Prediction**: Real-time revenue forecasting with adjustable parameters
- **Model Explainability**: SHAP waterfall plots and feature importance analysis  
- **Performance Analytics**: Cross-validation metrics and residual analysis
- **Data Visualisation**: Plotly charts for model comparison and insights

## 📊 Results

| Model         | R² Score | RMSE | Key Strengths                                   |
|---------------|----------|------|-------------------------------------------------|
| Random Forest | 0.87     | 0.45 | Feature interpretability, robust to outliers    |
| LightGBM      | 0.89     | 0.42 | Higher accuracy, efficient training             |

## 🛠 Tech Stack

**Core ML**: Python, scikit-learn, RandomForest, LightGBM, Pandas, NumPy  
**Visualisation**: Streamlit, Plotly, Matplotlib  
**Explainability**: SHAP  
**Data**: Full TMDB Movies Dataset 2024 (1M Movies) (Kaggle)

## 📁 Project Structure

```
ShowbizPredictor/
├── .venv/                              # Virtual environment
├── data/
│   ├── models/                         # Trained model artifacts (.pkl files)
│   └── tmdb_movie_dataset.csv          # TMDB dataset
├── notebooks/                          # Jupyter notebooks for exploration
│   ├── archive/
│   │   └── exploration_archive.ipynb
│   └── final/
│       ├── phase3_testing.ipynb
│       └── phase4_testing.ipynb
├── pages/                              # Streamlit multi-page app
│   ├── 1_🎯_Make_Prediction.py         # Interactive revenue prediction
│   ├── 2_🧠_SHAP_Analysis.py           # Model explainability dashboard
│   ├── 3_📊_Model_Comparison.py        # Performance comparison
│   └── 4_📈_Data_Insights.py           # Dataset analysis & visualisation
├── src/                                # Core ML modules
│   ├── __init__.py
│   ├── evaluate.py                     # Model evaluation metrics
│   ├── feature_engineering.py          # Feature creation pipeline
│   ├── preprocess.py                   # Data cleaning & preparation
│   └── train_model.py                  # Model training functions
├── .gitignore
├── 0_🏠_Home.py                        # Streamlit landing page
├── README.md
├── requirements.txt
├── train_all_models.py                 # Main training script
└── utils.py                            # Streamlit utilities & helper functions
```

## 🚀 Quick Start

1. **Clone repository**

   ```bash
   git clone https://github.com/yourusername/ShowbizPredictor
   cd ShowbizPredictor
   ```

2. **Setup environment**

   **macOS / Linux**
   
   ```bash
   python3 -m venv venv
   source venv/bin/activate
      ```
    
   **Windows**
   
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download [TMDB Movies Dataset (2023, 930k movies) on Kaggle](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)
 and place in data/**

5. **Train models**

   ```bash
   python train_all_models.py
   ```

6. **Launch application**

   ```bash
   streamlit run 0_🏠_Home.py
   ```

## 💡 Key Learning Outcomes

- **MLOps Pipeline**: Experienced full model lifecycle from data ingestion to deployment
- **Feature Engineering**: Applied advanced techniques (TF-IDF, SVD) to extract predictive signals from text data
- **Model Interpretability**: Implemented SHAP for explainable AI in business contexts
- **Software Engineering**: Structured codebase with separation of concerns and reusable modules
- **Interactive Dashboards**: Built professional web application with multiple analysis modes

## 🔮 Future Enhancements

- Model versioning and experiment tracking (MLflow)
- Automated model retraining pipeline
- Docker containerisation for deployment
- A/B testing framework for model comparison
- API endpoint for programmatic access

This project demonstrates the practical application of machine learning to solve real-world business problems while maintaining production-ready code quality and documentation standards.
