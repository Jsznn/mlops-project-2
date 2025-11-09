# MLOps Project

This repository contains a complete MLOps project for predicting customer subscription likelihood using machine learning models. The project includes model training, MLflow tracking, API service for model serving, and containerization.

## Project Structure

- `api_service.py`: FastAPI service for model predictions with caching and error handling
- `main.ipynb`: Main Jupyter notebook containing data analysis and model development
- `train_and_log_lr.py`: Script for training and logging the LogisticRegression model
- `mlflow_log_model.py`: Utilities for logging models to MLflow
- `preprocessing.py`: Data preprocessing utilities
- `test_api.py`: API service tests
- `test_customer_profiles.py`: Test cases for different customer profiles
- `Dockerfile`: Container definition for the API service
- `docker-compose.yml`: Service orchestration configuration
- `data/`: Directory containing dataset files
- `models/`: Saved model files
- `mlruns/`: MLflow tracking information

## Features

- Machine Learning Models:
  - Random Forest Classifier
  - Logistic Regression
- MLflow Integration:
  - Model versioning
  - Experiment tracking
  - Metrics logging
- FastAPI Service:
  - Model prediction endpoints
  - Request validation
  - Error handling
  - Response caching
  - CORS and Gzip middleware
- Docker Support:
  - Containerized API service
  - Multi-stage builds
  - Environment configuration

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Jsznn/mlops-project-2.git
   cd mlops-project-2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the API service:
   ```bash
   python api_service.py
   ```

   Or using Docker:
   ```bash
   docker-compose up
   ```

## API Usage

The API provides endpoints for making predictions using both Random Forest and Logistic Regression models:

### Endpoints

- `GET /`: Root endpoint with API information
- `GET /models`: List available models and their versions
- `POST /predict/best_rf`: Get predictions from Random Forest model
- `POST /predict/best_lr`: Get predictions from Logistic Regression model

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/best_rf" \
  -H "Content-Type: application/json" \
  -d '{
    "usia": 24,
    "pekerjaan": "manajer",
    "status_perkawinan": "lajang",
    "pendidikan": "Pendidikan Tinggi",
    "gagal_bayar_sebelumnya": "no",
    "pinjaman_rumah": "yes",
    "pinjaman_pribadi": "no",
    "jenis_kontak": "cellular",
    "bulan_kontak_terakhir": "jul",
    "hari_kontak_terakhir": "fri",
    "jumlah_kontak_kampanye_ini": 2,
    "hari_sejak_kontak_sebelumnya": 999,
    "jumlah_kontak_sebelumnya": 0,
    "hasil_kampanye_sebelumnya": "nonexistent",
    "tingkat_variasi_pekerjaan": -1.7,
    "indeks_harga_konsumen": 94.215,
    "indeks_kepercayaan_konsumen": -40.3,
    "suku_bunga_euribor_3bln": 0.885,
    "jumlah_pekerja": 4991.6,
    "pulau": "Papua"
  }'
```

### Example Response

```json
{
  "model_name": "best_rf",
  "prediction": 1,
  "probability_subscribe": 0.7555,
  "model_version": "local"
}
```

## Development

- Run tests: `python -m pytest test_api.py test_customer_profiles.py`
- Check API documentation: Visit `http://localhost:8000/docs` when API is running
- Monitor MLflow: Start MLflow UI with `mlflow ui`

## License

MIT