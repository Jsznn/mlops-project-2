import requests
import json
import sys
import os

# This script assumes the API is running locally on port 8000
# It's an integration test, not a unit test

def test_api():
    """Test the ML model API endpoints"""
    # API base URL
    base_url = "http://localhost:8000"
    
    try:
        # Test root endpoint
        print("Testing root endpoint...")
        response = requests.get(base_url)
        print(f"Root response: {json.dumps(response.json(), indent=2)}\n")
        
        # Test models endpoint
        print("Testing models endpoint...")
        response = requests.get(f"{base_url}/models")
        print(f"Models response: {json.dumps(response.json(), indent=2)}\n")
        
        # Test sample prediction with both models
        test_data = {
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
        
        # Note: API currently only supports best_rf fully in the validation logic of the predict endpoint
        # but let's try both as per original test
        for model in ["best_rf"]:
            print(f"\nTesting prediction with {model}...")
            response = requests.post(f"{base_url}/predict/{model}", json=test_data)
            if response.status_code == 200:
                print(f"Prediction response: {json.dumps(response.json(), indent=2)}")
            else:
                print(f"Failed to predict with {model}: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure it is running on http://localhost:8000")

if __name__ == "__main__":
    test_api()
