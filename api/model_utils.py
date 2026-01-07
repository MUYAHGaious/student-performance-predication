"""
Model Utilities Module

Handles model loading and prediction logic.
"""

import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class ModelPredictor:
    """Wrapper class for model prediction with preprocessing."""

    def __init__(self, model_path: str):
        """
        Initialize the predictor by loading the model package.

        Args:
            model_path: Path to the saved model file (.pkl or .joblib)
        """
        self.model_path = Path(model_path)
        self.model_package = self._load_model()

        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.feature_cols = self.model_package['feature_cols']
        self.model_name = self.model_package['model_name']
        self.metrics = self.model_package.get('metrics', {})

        print(f"✓ Model loaded: {self.model_name}")
        print(f"✓ Features: {len(self.feature_cols)}")
        print(f"✓ Test F1-Score: {self.metrics.get('f1_score', 'N/A')}")

    def _load_model(self) -> Dict[str, Any]:
        """Load model package from file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # Support both .pkl and .joblib
        if self.model_path.suffix == '.joblib':
            return joblib.load(self.model_path)
        else:
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)

    def preprocess_input(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data to match model expectations.

        Args:
            input_data: Dictionary with student features

        Returns:
            DataFrame ready for prediction
        """
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all required features are present
        missing_features = set(self.feature_cols) - set(df.columns)
        if missing_features:
            # Fill missing features with default values (0)
            for feat in missing_features:
                df[feat] = 0

        # Select only required features in correct order
        df = df[self.feature_cols]

        # Scale features
        df_scaled = self.scaler.transform(df)
        df_scaled = pd.DataFrame(df_scaled, columns=self.feature_cols)

        return df_scaled

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction on input data.

        Args:
            input_data: Dictionary with student features

        Returns:
            Dictionary with prediction and confidence scores
        """
        # Preprocess
        X = self.preprocess_input(input_data)

        # Predict
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Format response
        result = {
            "prediction": "at_risk" if prediction == 1 else "on_track",
            "confidence": float(max(probabilities)),
            "probability_scores": {
                "on_track": float(probabilities[0]),
                "at_risk": float(probabilities[1])
            },
            "model_name": self.model_name,
            "input_features": input_data
        }

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance metrics."""
        return {
            "model_name": self.model_name,
            "n_features": len(self.feature_cols),
            "feature_names": self.feature_cols,
            "performance_metrics": self.metrics
        }


def validate_input_data(data: Dict[str, Any], required_features: List[str]) -> tuple:
    """
    Validate input data structure.

    Args:
        data: Input data dictionary
        required_features: List of required feature names

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if data is a dictionary
    if not isinstance(data, dict):
        return False, "Input must be a JSON object"

    # Check for empty data
    if not data:
        return False, "Input data cannot be empty"

    # Check data types for numerical features
    for key, value in data.items():
        if not isinstance(value, (int, float, list)):
            return False, f"Feature '{key}' must be numeric (int/float) or list"

    return True, None
