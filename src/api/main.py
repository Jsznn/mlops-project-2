import os
import logging
import sys
from typing import Dict, Optional, List
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import mlflow
import uvicorn
from contextlib import asynccontextmanager

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config, load_mappings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configs
CONFIG = load_config()
MAPPINGS = load_mappings()

# Global model cache
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load RandomForest model during startup
        logger.info("Loading RandomForest model...")
        model_path = CONFIG['paths']['model_rf']
        # Handle path relative to root
        if not os.path.exists(model_path):
             # Try going up levels if running from src/api
             if os.path.exists(os.path.join('..', '..', model_path)):
                 model_path = os.path.join('..', '..', model_path)
        
        models['best_rf'] = joblib.load(model_path)
        logger.info("Successfully loaded RandomForest model")
        yield
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    finally:
        # Clean up resources during shutdown
        models.clear()

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing ML Models API",
    description="API for serving predictions from Bank Marketing ML models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Input data model
class CustomerData(BaseModel):
    usia: int = Field(..., description="Age of the customer")
    pekerjaan: str = Field(..., description="Job/occupation")
    status_perkawinan: str = Field(..., description="Marital status")
    pendidikan: str = Field(..., description="Education level")
    gagal_bayar_sebelumnya: str = Field(..., description="Has credit default")
    pinjaman_rumah: str = Field(..., description="Has housing loan")
    pinjaman_pribadi: str = Field(..., description="Has personal loan")
    jenis_kontak: str = Field(..., description="Contact type")
    bulan_kontak_terakhir: str = Field(..., description="Last contact month")
    hari_kontak_terakhir: str = Field(..., description="Last contact day")
    jumlah_kontak_kampanye_ini: int = Field(..., description="Number of contacts in this campaign")
    hari_sejak_kontak_sebelumnya: int = Field(..., description="Days since last contact")
    jumlah_kontak_sebelumnya: int = Field(..., description="Number of previous contacts")
    hasil_kampanye_sebelumnya: str = Field(..., description="Outcome of previous campaign")
    tingkat_variasi_pekerjaan: float = Field(..., description="Employment variation rate")
    indeks_harga_konsumen: float = Field(..., description="Consumer price index")
    indeks_kepercayaan_konsumen: float = Field(..., description="Consumer confidence index")
    suku_bunga_euribor_3bln: float = Field(..., description="3 month euribor rate")
    jumlah_pekerja: float = Field(..., description="Number of employees")
    pulau: str = Field(..., description="Island/region")

    class Config:
        schema_extra = {
            "example": {
                "usia": 41,
                "pekerjaan": "sosial media specialis",
                "status_perkawinan": "menikah",
                "pendidikan": "Pendidikan Tinggi",
                "gagal_bayar_sebelumnya": "no",
                "pinjaman_rumah": "yes",
                "pinjaman_pribadi": "no",
                "jenis_kontak": "cellular",
                "bulan_kontak_terakhir": "may",
                "hari_kontak_terakhir": "fri",
                "jumlah_kontak_kampanye_ini": 2,
                "hari_sejak_kontak_sebelumnya": 999,
                "jumlah_kontak_sebelumnya": 0,
                "hasil_kampanye_sebelumnya": "nonexistent",
                "tingkat_variasi_pekerjaan": -1.8,
                "indeks_harga_konsumen": 92.893,
                "indeks_kepercayaan_konsumen": -46.2,
                "suku_bunga_euribor_3bln": 1.244,
                "jumlah_pekerja": 5099.1,
                "pulau": "Jawa"
            }
        }

# Prediction response model
class PredictionResponse(BaseModel):
    model_name: str
    prediction: int
    probability_subscribe: float
    model_version: Optional[str] = None

def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """Convert input data to DataFrame and apply preprocessing"""
    # Convert to dataframe
    df = pd.DataFrame([data.dict()])
    
    # Calculate high rate flag
    mean_rate = CONFIG['preprocessing']['mean_euribor_rate']
    df['high_rate_flag'] = (df['suku_bunga_euribor_3bln'] > mean_rate).astype(int)
    
    # Map job sectors
    job_groups = MAPPINGS['job_sector_groups']
    formal = set(job_groups['formal'])
    informal = set(job_groups['informal'])
    non_employed = set(job_groups['non_employed'])
    
    def map_sector(job):
        if job in formal:
            return 'formal'
        elif job in informal:
            return 'informal'
        elif job in non_employed:
            return 'non_employed'
        return 'other'
    
    df['job_sector'] = df['pekerjaan'].apply(map_sector)
    df['job_sector'] = df['job_sector'].map(MAPPINGS['job_sector'])
    
    # Apply category mappings
    for col, mapping in MAPPINGS.items():
        if col in df.columns and col != 'job_sector_groups':
            df[col] = df[col].map(mapping)
    
    return df

# Cached model getter
@lru_cache(maxsize=None)
def get_model(model_name: str):
    """Get model from cache"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return models[model_name]

def load_model(model_name: str):
    """Load model from in-memory cache, MLflow, or local file and return (model, version)."""
    # 1) Check in-memory models loaded at startup
    if model_name in models:
        return models[model_name], "local"

    # 2) Try to get from MLflow (if available)
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name)
        if versions:
            mv = versions[0]
            # try to load via mlflow's pyfunc loader using the version source or models:/ URI
            try:
                model_uri = getattr(mv, "source", f"models:/{model_name}/{mv.version}")
                model = mlflow.pyfunc.load_model(model_uri)
                return model, mv.version
            except Exception:
                # fallback: continue to local loading
                pass
    except Exception:
        # ignore MLflow errors and try local
        pass

    # 3) Try local joblib file
    # Construct path based on config or convention
    local_path = os.path.join('models', f'{model_name}.joblib')
    if not os.path.exists(local_path):
         # Try going up levels
         if os.path.exists(os.path.join('..', '..', local_path)):
             local_path = os.path.join('..', '..', local_path)
             
    if os.path.exists(local_path):
        model = joblib.load(local_path)
        return model, "local"

    # If not found anywhere, raise HTTPException
    raise HTTPException(status_code=404, detail=f"Model {model_name} not found")


@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "message": "Bank Marketing ML Models API",
        "models_available": list(models.keys()),
        "docs_url": "/docs"
    }

@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, customer_data: CustomerData):
    """Make prediction using specified model"""
    logger.info(f"Received prediction request for model: {model_name}")
    
    if model_name != "best_rf":
        logger.error(f"Invalid model name requested: {model_name}")
        raise HTTPException(status_code=400, detail="Currently only 'best_rf' model is supported")
    
    try:
        # Preprocess input
        logger.info("Preprocessing input data...")
        df = preprocess_input(customer_data)
        
        # Load model
        logger.info(f"Loading model {model_name}...")
        model, version = load_model(model_name)
        
        # Make prediction
        logger.info("Making prediction...")
        
        # Handle both sklearn models and MLflow models
        if hasattr(model, 'predict'):
            prediction = model.predict(df)[0]
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(df)[0][1]  # Probability of class 1
            else:
                # For models without probability support, use decision function if available
                if hasattr(model, 'decision_function'):
                    decision = model.decision_function(df)[0]
                    prob = 1 / (1 + np.exp(-decision))  # Sigmoid transformation
                else:
                    prob = float(prediction)  # Fallback to prediction as probability
        else:
            # MLflow model
            prediction = model.predict(df)
            if isinstance(prediction, pd.DataFrame):
                prediction = prediction.iloc[0]
            if len(prediction) > 1:
                # If prediction includes probabilities
                prob = prediction[1]
                prediction = int(prediction[0])
            else:
                prediction = int(prediction[0])
                prob = float(prediction)
                
        logger.info(f"Prediction complete: {prediction} (probability: {prob:.2f})")
        
        return PredictionResponse(
            model_name=model_name,
            prediction=int(prediction),
            probability_subscribe=float(prob),
            model_version=version
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/models")
def list_models():
    """List available models and their versions"""
    model_list = []
    
    # Check MLflow models
    try:
        client = mlflow.tracking.MlflowClient()
        for name in ["best_rf", "best_lr"]:
            try:
                versions = client.get_latest_versions(name)
                if versions:
                    model_list.append({
                        "name": name,
                        "source": "mlflow",
                        "version": versions[0].version,
                        "status": "available"
                    })
                    continue
            except Exception:
                pass
            
            # Check local files
            local_path = os.path.join('models', f'{name}.joblib')
            if not os.path.exists(local_path):
                 if os.path.exists(os.path.join('..', '..', local_path)):
                     local_path = os.path.join('..', '..', local_path)

            if os.path.exists(local_path):
                model_list.append({
                    "name": name,
                    "source": "local",
                    "version": "local",
                    "status": "available"
                })
            else:
                model_list.append({
                    "name": name,
                    "source": None,
                    "version": None,
                    "status": "not_found"
                })
    except Exception as e:
        return {"error": str(e), "models_found": model_list}
    
    return {"models": model_list}

if __name__ == "__main__":
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
