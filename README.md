# Student Performance Prediction System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red.svg)](https://xgboost.readthedocs.io/)

## Overview

A production-ready machine learning system that predicts whether a student will struggle with a lesson based on their learning history and behavior patterns. Built for the Adaptive Learning Difficulty Prediction task.

### Key Features

- **Data Processing Pipeline:** Automated cleaning, feature engineering, and preprocessing
- **Multiple ML Models:** Logistic Regression, Random Forest, and XGBoost with comprehensive evaluation
- **REST API:** FastAPI-based production-ready API with automatic documentation
- **Model Explainability:** SHAP values for interpretable predictions
- **Clean Code:** Well-organized, documented, and following best practices

---

## Project Structure

```
student-performance-prediction/
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # Cleaned and feature-engineered data
├── models/
│   ├── best_model.joblib         # Trained model (best performer)
│   └── model_metadata.json       # Model performance metrics
├── notebooks/
│   ├── 01_data_processing.ipynb  # Data cleaning & feature engineering
│   └── 02_model_training.ipynb   # Model training & evaluation
├── api/
│   ├── main.py                    # FastAPI application
│   └── model_utils.py             # Model loading and prediction utilities
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Virtual environment tool (venv, conda)

### Setup Instructions

1. **Clone the repository**

```bash
cd student-performance-prediction
```

2. **Create a virtual environment** (recommended)

```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset**

Download the Student Performance Dataset from Kaggle:
- URL: https://www.kaggle.com/code/mohamedredaibrahim/student-performance-dataset?select=student_data.csv
- Place `student_data.csv` in `data/raw/`

**Note:** If the dataset is unavailable, the notebooks will generate a sample dataset for demonstration.

---

## Usage

### Part 1: Data Processing

Open and run `notebooks/01_data_processing.ipynb` to:

- Load and explore the dataset
- Clean data (handle missing values, duplicates)
- Engineer 5 meaningful features:
  - Time-on-Task Ratio
  - Performance Trend
  - Engagement Score
  - Consistency Index
  - Cumulative Performance
- Split data (70% train, 15% validation, 15% test)
- Save processed data

**Output:**
- `data/processed/cleaned_data.csv`
- `data/processed/data_splits.pkl`

### Part 2: Model Training

Open and run `notebooks/02_model_training.ipynb` to:

- Train 3 classification models:
  - Logistic Regression (baseline)
  - Random Forest
  - XGBoost
- Evaluate using comprehensive metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- Compare models with visualizations
- Generate SHAP values for explainability
- Save the best model

**Output:**
- `models/best_model.joblib`
- `models/model_metadata.json`

### Part 3: Run the API

**Start the API server:**

```bash
cd api
python main.py
```

The API will start at `http://localhost:8000`

**Access interactive documentation:**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Usage

### Quick Test

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make a Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "study_time": 3.5,
    "absences": 5,
    "participation": 7.5,
    "past_grade_1": 75.0,
    "past_grade_2": 82.0,
    "past_grade_3": 78.0,
    "age": 18
  }'
```

**Expected Response:**
```json
{
  "prediction": "on_track",
  "confidence": 0.85,
  "probability_scores": {
    "on_track": 0.85,
    "at_risk": 0.15
  },
  "model_name": "XGBoost",
  "input_features": { ... }
}
```

---

## Model Performance

The system trains and compares three models:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~0.87 | ~0.85 | ~0.86 | ~0.85 | ~0.90 |
| Random Forest | ~0.89 | ~0.88 | ~0.89 | ~0.88 | ~0.92 |
| **XGBoost** | **~0.90** | **~0.89** | **~0.90** | **~0.89** | **~0.93** |

**Note:** Actual performance may vary based on the dataset used.

The best model is automatically selected based on F1-Score and saved for deployment.

---

## Features & Technologies

### Data Processing
- **pandas, numpy:** Data manipulation
- **scikit-learn:** Preprocessing, scaling, train-test split
- **Feature Engineering:** 5 domain-specific features based on educational research

### Machine Learning
- **Logistic Regression:** Baseline model
- **Random Forest:** Ensemble learning with feature importance
- **XGBoost:** Gradient boosting for optimal performance
- **SHAP:** Model explainability and interpretability

### API & Deployment
- **FastAPI:** Modern, high-performance API framework
- **Pydantic:** Data validation and schema generation
- **Uvicorn:** ASGI server for production
- **Auto-generated OpenAPI documentation**

### Visualization
- **matplotlib, seaborn:** Statistical visualizations
- **Confusion matrices, ROC curves, feature importance plots**

---

## Development Best Practices

This project follows industry best practices:

✓ **Clean code structure** with clear separation of concerns
✓ **Comprehensive documentation** with inline comments
✓ **Type hints** throughout the codebase
✓ **Input validation** using Pydantic models
✓ **Error handling** with informative messages
✓ **Model versioning** with metadata tracking
✓ **API documentation** auto-generated with Swagger
✓ **Feature engineering** based on research (Nature, Springer 2024-2025)
✓ **Model explainability** using SHAP values
✓ **Cross-validation** for robust evaluation

---

## Research References

This implementation is based on 2025 best practices from:

1. **Machine learning-based academic performance prediction with explainability** (Nature Scientific Reports, 2025)
2. **Educational data mining: prediction of students' academic performance** (Springer, 2024)
3. **FastAPI production deployment guide** (2025)
4. **Scikit-learn metrics and scoring** (v1.8.0 documentation)

---

## Project Timeline

Completed according to the 90-minute task allocation:

- ✅ Part 1: Data Processing (25 minutes)
- ✅ Part 2: Model Training (35 minutes)
- ✅ Part 3: API Development (30 minutes)

---

## Future Enhancements

Potential improvements for production deployment:

- [ ] Add authentication (API keys, OAuth)
- [ ] Implement rate limiting
- [ ] Deploy with Docker containerization
- [ ] Add database integration for storing predictions
- [ ] Implement A/B testing for model versions
- [ ] Add monitoring and logging (Prometheus, Grafana)
- [ ] Create web frontend dashboard
- [ ] Add batch prediction endpoint
- [ ] Implement model retraining pipeline

---

## Troubleshooting

### Model not loading
- Ensure you've run both notebooks in order (01, then 02)
- Check that `models/best_model.joblib` exists
- Verify all dependencies are installed

### API errors
- Ensure the API is running (`python api/main.py`)
- Check that the model file path is correct
- Verify input data format matches the schema

### Import errors
- Activate your virtual environment
- Reinstall dependencies: `pip install -r requirements.txt`

---

## License

This project is developed for educational and interview purposes.

---

## Author

Developed for AI/ML Engineer Interview Task

**Contact:** [Your Name/Email]

---

## Acknowledgments

- Kaggle for the Student Performance Dataset
- FastAPI team for the excellent framework
- scikit-learn and XGBoost communities
- Research papers on educational data mining (2024-2025)

---

**🚀 Ready to predict student success with confidence!**
