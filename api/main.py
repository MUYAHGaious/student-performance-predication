"""
Student Performance Prediction API

FastAPI application for predicting student learning difficulty.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
import uvicorn
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.model_utils import ModelPredictor, validate_input_data

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model.joblib"
API_VERSION = "1.0.0"
API_TITLE = "Student Performance Prediction API"

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title=API_TITLE,
    description="""
    ## Adaptive Learning Difficulty Prediction API

    This API predicts whether a student is at risk of struggling with a lesson
    based on their learning history and behavior patterns.

    ### Features:
    - **Real-time predictions** with confidence scores
    - **Comprehensive input validation**
    - **Model explainability** with probability scores
    - **Health monitoring** endpoint

    ### Model Information:
    The prediction model uses machine learning algorithms trained on student
    performance data, including study time, grades, attendance, and engagement metrics.
    """,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# CORS Middleware
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Load Model
# ============================================================================

try:
    predictor = ModelPredictor(str(MODEL_PATH))
    print(f"✓ API initialized with model: {predictor.model_name}")
except Exception as e:
    print(f"ERROR: Failed to load model: {e}")
    predictor = None

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class StudentFeatures(BaseModel):
    """Input features for student performance prediction."""

    # Core academic features
    study_time: float = Field(
        ...,
        ge=0,
        le=10,
        description="Weekly study time in hours",
        example=3.5
    )
    absences: int = Field(
        ...,
        ge=0,
        le=100,
        description="Number of absences",
        example=5
    )
    participation: float = Field(
        ...,
        ge=0,
        le=10,
        description="Participation score (0-10)",
        example=7.5
    )

    # Past grades (optional, will be computed if not provided)
    past_grade_1: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="First past grade score",
        example=75.0
    )
    past_grade_2: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Second past grade score",
        example=82.0
    )
    past_grade_3: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Third past grade score",
        example=78.0
    )

    # Additional features
    age: Optional[int] = Field(
        None,
        ge=10,
        le=100,
        description="Student age",
        example=18
    )

    # Engineered features (optional, will be auto-computed)
    time_on_task_ratio: Optional[float] = None
    performance_trend: Optional[float] = None
    engagement_score: Optional[float] = None
    consistency_index: Optional[float] = None
    cumulative_performance: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "study_time": 3.5,
                "absences": 5,
                "participation": 7.5,
                "past_grade_1": 75.0,
                "past_grade_2": 82.0,
                "past_grade_3": 78.0,
                "age": 18
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    prediction: str = Field(
        ...,
        description="Predicted student status: 'at_risk' or 'on_track'"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence score of the prediction (0-1)"
    )
    probability_scores: Dict[str, float] = Field(
        ...,
        description="Probability scores for each class"
    )
    model_name: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    input_features: Dict[str, Any] = Field(
        ...,
        description="Input features used for prediction"
    )

    class Config:
        schema_extra = {
            "example": {
                "prediction": "on_track",
                "confidence": 0.85,
                "probability_scores": {
                    "on_track": 0.85,
                    "at_risk": 0.15
                },
                "model_name": "XGBoost",
                "input_features": {
                    "study_time": 3.5,
                    "absences": 5,
                    "participation": 7.5
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: Optional[str]
    api_version: str


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str
    error_type: str

# ============================================================================
# API Endpoints
# ============================================================================

@app.get(
    "/",
    summary="API Root",
    description="Welcome endpoint with API information"
)
async def root():
    """Root endpoint."""
    return {
        "message": "Student Performance Prediction API",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API and model are running properly"
)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "model_loaded": predictor is not None,
        "model_name": predictor.model_name if predictor else None,
        "api_version": API_VERSION
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Make Prediction",
    description="Predict if a student is at risk based on their features",
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input data", "model": ErrorResponse},
        500: {"description": "Internal server error", "model": ErrorResponse}
    }
)
async def predict(student_features: StudentFeatures):
    """
    Make a prediction for student performance.

    **Input:** Student features including study time, absences, participation, and past grades.

    **Output:** Prediction result with confidence score.
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded. Please contact administrator."
        )

    try:
        # Convert to dictionary
        input_data = student_features.dict(exclude_none=False)

        # Validate input
        is_valid, error_msg = validate_input_data(input_data, predictor.feature_cols)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )

        # Make prediction
        result = predictor.predict(input_data)

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get(
    "/model/info",
    summary="Model Information",
    description="Get information about the loaded model and its performance"
)
async def get_model_info():
    """Get model metadata and performance metrics."""
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model not loaded"
        )

    try:
        return predictor.get_model_info()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc)
    )

# ============================================================================
# Run Application
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Starting Student Performance Prediction API")
    print("="*60)
    print(f"API Version: {API_VERSION}")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Docs: http://localhost:8000/docs")
    print("="*60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
