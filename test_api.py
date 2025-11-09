import requests
import json

def test_api():
    """Test the ML model API endpoints"""
    # API base URL
    base_url = "http://localhost:8000"
    
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
    
    for model in ["best_rf", "best_lr"]:
        print(f"\nTesting prediction with {model}...")
        response = requests.post(f"{base_url}/predict/{model}", json=test_data)
        print(f"Prediction response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    test_api()