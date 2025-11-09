import os
import logging
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load RandomForest model during startup
        logger.info("Loading RandomForest model...")
        models['best_rf'] = joblib.load('models/best_rf.joblib')
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

# Load preprocessing mappings (copy from preprocessing.py for inference)
CATEGORY_MAPPINGS = {
    'pekerjaan': {
        'sosial media specialis': 0, 'teknisi': 1, 'pekerja kasar': 2,
        'manajer': 3, 'asisten rumah tangga': 4, 'mahasiswa': 5,
        'penyedia jasa': 6, 'pemilik bisnis': 7, 'entrepreneur': 8,
        'pengangguran': 9, 'pensiunan': 10, 'unknown': 11
    },
    'status_perkawinan': {
        'menikah': 0, 'lajang': 1, 'cerai': 2, 'unknown': 3
    },
    'pendidikan': {
        'TIDAK SEKOLAH': 0, 'Tidak Tamat SD': 1, 'SD': 2, 'SMP': 3,
        'SMA': 4, 'Diploma': 5, 'Pendidikan Tinggi': 6, 'unknown': 7
    },
    'gagal_bayar_sebelumnya': {'no': 0, 'yes': 1},
    'pinjaman_rumah': {'no': 0, 'yes': 1, 'unknown': 2},
    'pinjaman_pribadi': {'no': 0, 'yes': 1, 'unknown': 2},
    'jenis_kontak': {'cellular': 0, 'telephone': 1},
    'bulan_kontak_terakhir': {
        'mar': 0, 'apr': 1, 'may': 2, 'jun': 3, 'jul': 4,
        'aug': 5, 'sep': 6, 'oct': 7, 'nov': 8, 'dec': 9
    },
    'hari_kontak_terakhir': {
        'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5
    },
    'hasil_kampanye_sebelumnya': {
        'nonexistent': 0, 'failure': 1, 'success': 2
    },
    'pulau': {
        'Jawa': 0, 'Sumatera': 1, 'Kalimantan': 2, 'Sulawesi': 3,
        'Bali': 4, 'NTB': 5, 'NTT': 6, 'Papua': 7
    }
}

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
    mean_rate = 4.5  # Approximate from training data
    df['high_rate_flag'] = (df['suku_bunga_euribor_3bln'] > mean_rate).astype(int)
    
    # Map job sectors
    formal = {'manajer', 'pemilik bisnis', 'entrepreneur', 'teknisi', 'sosial media specialis'}
    informal = {'pekerja kasar', 'asisten rumah tangga', 'penyedia jasa'}
    non_employed = {'mahasiswa', 'pengangguran', 'pensiunan', 'unknown'}
    
    def map_sector(job):
        if job in formal:
            return 'formal'
        elif job in informal:
            return 'informal'
        elif job in non_employed:
            return 'non_employed'
        return 'other'
    
    df['job_sector'] = df['pekerjaan'].apply(map_sector)
    df['job_sector'] = df['job_sector'].map({'formal': 0, 'informal': 1, 'non_employed': 2, 'other': 3})
    
    # Apply category mappings
    for col, mapping in CATEGORY_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

# Cached model getter
@lru_cache(maxsize=None)
def get_model(model_name: str):
    """Get model from cache"""
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return models[model_name]

# Cached preprocessing
@lru_cache(maxsize=1000)
def _calculate_mean_rate():
    """Cache the mean rate calculation"""
    return 4.857  # Precalculated mean from training data

@lru_cache(maxsize=1000)
def _map_job_sector(job: str) -> int:
    """Cache job sector mapping"""
    formal = {'manajer', 'pemilik bisnis', 'entrepreneur', 'teknisi', 'sosial media specialis'}
    informal = {'pekerja kasar', 'asisten rumah tangga', 'penyedia jasa'}
    non_employed = {'mahasiswa', 'pengangguran', 'pensiunan', 'unknown'}
    
    if job in formal:
        return 0  # formal
    elif job in informal:
        return 1  # informal
    elif job in non_employed:
        return 2  # non_employed
    return 3  # other


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
    local_path = os.path.join('models', f'{model_name}.joblib')
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
    models = []
    
    # Check MLflow models
    try:
        client = mlflow.tracking.MlflowClient()
        for name in ["best_rf", "best_lr"]:
            try:
                versions = client.get_latest_versions(name)
                if versions:
                    models.append({
                        "name": name,
                        "source": "mlflow",
                        "version": versions[0].version,
                        "status": "available"
                    })
                    continue
            except Exception:
                pass
            
            # Check local files
            model_path = os.path.join('models', f'{name}.joblib')
            if os.path.exists(model_path):
                models.append({
                    "name": name,
                    "source": "local",
                    "version": "local",
                    "status": "available"
                })
            else:
                models.append({
                    "name": name,
                    "source": None,
                    "version": None,
                    "status": "not_found"
                })
    except Exception as e:
        return {"error": str(e), "models_found": models}
    
    return {"models": models}

if __name__ == "__main__":
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)