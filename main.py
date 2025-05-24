"""
FastAPI Restaurant Anomaly Detection API
=======================================

A REST API for predicting anomalies in restaurant operations.
Configured for deployment on Render with CORS enabled.

Endpoints:
- POST /predict - Single prediction
- POST /predict-batch - Multiple predictions
- GET /health - Health check
- GET /model-info - Model information
- GET / - API documentation

Usage:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import uvicorn
import os
import logging
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import our predictor class
# Note: Make sure restaurant_anomaly_predictor.py is in the same directory
try:
    from restaurant_anomaly_predictor import RestaurantAnomalyPredictor
except ImportError:
    print("Warning: restaurant_anomaly_predictor.py not found. Please ensure it's in the same directory.")
    RestaurantAnomalyPredictor = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown"""
    global predictor
    
    if RestaurantAnomalyPredictor is None:
        logger.error("RestaurantAnomalyPredictor not available")
        yield
        return
    
    try:
        # Initialize predictor
        predictor = RestaurantAnomalyPredictor()
        
        # Try to load the model
        model_path = os.getenv('MODEL_PATH', 'restaurant_anomaly_model.pkl')
        if os.path.exists(model_path):
            predictor.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found at {model_path}. API will run but predictions will fail.")
            
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        predictor = None
    
    yield  # Application runs here
    
    # Cleanup (if needed)
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Restaurant Anomaly Detection API",
    description="Detect anomalies in restaurant operations using machine learning",
    version="1.0.0",
    docs_url="/",  # Swagger UI at root
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for all origins (as requested)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    zone: str = Field(..., description="Zone identifier")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    occupancy: int = Field(..., ge=0, le=100, description="Number of people (0-100)")
    power_use: float = Field(..., ge=1.0, le=10.0, description="Power usage (1-10)")
    water_use: float = Field(..., ge=1.0, le=20.0, description="Water usage (1-20)")
    cleaning_status: str = Field(..., description="Cleaning status")
    
    @validator('zone')
    def validate_zone(cls, v):
        valid_zones = ['Store01', 'Dining01', 'Kitchen01', 'Hallway01']
        if v not in valid_zones:
            raise ValueError(f'Zone must be one of: {valid_zones}')
        return v
    
    @validator('cleaning_status')
    def validate_cleaning_status(cls, v):
        valid_statuses = ['pending', 'done', 'inprogress']
        if v not in valid_statuses:
            raise ValueError(f'Cleaning status must be one of: {valid_statuses}')
        return v

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: str = Field(..., description="Anomaly or Normal")
    anomaly_probability: float = Field(..., description="Probability of anomaly (0-1)")
    normal_probability: float = Field(..., description="Probability of normal (0-1)")
    risk_level: str = Field(..., description="Risk level: Low, Medium, or High")
    input_data: Dict[str, Any] = Field(..., description="Input data used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    timestamp: str = Field(..., description="Batch prediction timestamp")

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_loaded: bool = Field(..., description="Whether model is loaded")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    model_path: str = Field(..., description="Model file path")
    timestamp: str = Field(..., description="Info request timestamp")

# API Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None and predictor.is_trained,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model"""
    if predictor is None or not predictor.is_trained:
        return ModelInfoResponse(
            model_loaded=False,
            feature_importance=None,
            model_path=os.getenv('MODEL_PATH', 'restaurant_anomaly_model.pkl'),
            timestamp=datetime.now().isoformat()
        )
    
    try:
        feature_importance = predictor.get_feature_importance()
        return ModelInfoResponse(
            model_loaded=True,
            feature_importance=feature_importance,
            model_path=os.getenv('MODEL_PATH', 'restaurant_anomaly_model.pkl'),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return ModelInfoResponse(
            model_loaded=True,
            feature_importance=None,
            model_path=os.getenv('MODEL_PATH', 'restaurant_anomaly_model.pkl'),
            timestamp=datetime.now().isoformat()
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(request: PredictionRequest):
    """Make a single anomaly prediction"""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model file is available."
        )
    
    try:
        # Make prediction
        result = predictor.predict(
            zone=request.zone,
            hour=request.hour,
            occupancy=request.occupancy,
            power_use=request.power_use,
            water_use=request.water_use,
            cleaning_status=request.cleaning_status
        )
        
        # Format response
        return PredictionResponse(
            prediction=result['prediction'],
            anomaly_probability=result['anomaly_probability'],
            normal_probability=result['normal_probability'],
            risk_level=result['risk_level'],
            input_data=result['input_data'],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch", response_model=BatchPredictionResponse)
async def predict_batch_anomalies(request: BatchPredictionRequest):
    """Make multiple anomaly predictions"""
    if predictor is None or not predictor.is_trained:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please ensure the model file is available."
        )
    
    if len(request.predictions) > 1000:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size cannot exceed 1000 predictions"
        )
    
    try:
        # Prepare data for batch prediction
        batch_data = []
        for pred_request in request.predictions:
            batch_data.append([
                pred_request.zone,
                pred_request.hour,
                pred_request.occupancy,
                pred_request.power_use,
                pred_request.water_use,
                pred_request.cleaning_status
            ])
        
        # Make batch predictions
        results = predictor.predict_batch(batch_data)
        
        # Format responses
        predictions = []
        anomaly_count = 0
        
        for result in results:
            predictions.append(PredictionResponse(
                prediction=result['prediction'],
                anomaly_probability=result['anomaly_probability'],
                normal_probability=result['normal_probability'],
                risk_level=result['risk_level'],
                input_data=result['input_data'],
                timestamp=datetime.now().isoformat()
            ))
            
            if result['prediction'] == 'Anomaly':
                anomaly_count += 1
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            anomaly_count=anomaly_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/examples")
async def get_prediction_examples():
    """Get example requests for testing the API"""
    return {
        "single_prediction_example": {
            "zone": "Kitchen01",
            "hour": 14,
            "occupancy": 25,
            "power_use": 8.5,
            "water_use": 16.2,
            "cleaning_status": "pending"
        },
        "batch_prediction_example": {
            "predictions": [
                {
                    "zone": "Dining01",
                    "hour": 12,
                    "occupancy": 45,
                    "power_use": 4.5,
                    "water_use": 8.2,
                    "cleaning_status": "done"
                },
                {
                    "zone": "Store01",
                    "hour": 14,
                    "occupancy": 15,
                    "power_use": 8.5,
                    "water_use": 16.8,
                    "cleaning_status": "pending"
                }
            ]
        }
    }

# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# For local development
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="127.0.0.0",
        port=port,
        reload=False,  # Set to False for production
        log_level="info"
    )