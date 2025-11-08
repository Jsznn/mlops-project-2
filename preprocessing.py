import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(input_path, output_path):
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Handle 'gagal_bayar_sebelumnya' column
    df['gagal_bayar_sebelumnya'] = df['gagal_bayar_sebelumnya'].apply(lambda x: 'no' if x == 'no' else 'yes')
    
    # Feature Engineering
    # High-Rate Flag
    mean_rate = df['suku_bunga_euribor_3bln'].mean()
    df['high_rate_flag'] = (df['suku_bunga_euribor_3bln'] > mean_rate).astype(int)
    
    # Sector Grouping
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
        else:
            return 'other'

    df['job_sector'] = df['pekerjaan'].apply(map_sector)
    
    # Category Mappings
    mappings = {
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
        },
        'job_sector': {'formal': 0, 'informal': 1, 'non_employed': 2}
    }
    
    # Apply all mappings
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)
    
    # Save preprocessed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    input_path = "data/dataset.csv"
    output_path = "data/preprocessed.csv"
    df = preprocess_data(input_path, output_path)