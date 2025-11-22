import pandas as pd
import numpy as np
import warnings
import os
import sys

# Add project root to path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.config_loader import load_config, load_mappings

warnings.filterwarnings('ignore')

def preprocess_data(input_path, output_path):
    # Load mappings
    mappings = load_mappings()
    
    # Read the dataset
    df = pd.read_csv(input_path)
    
    # Handle 'gagal_bayar_sebelumnya' column
    df['gagal_bayar_sebelumnya'] = df['gagal_bayar_sebelumnya'].apply(lambda x: 'no' if x == 'no' else 'yes')
    
    # Feature Engineering
    # High-Rate Flag
    mean_rate = df['suku_bunga_euribor_3bln'].mean()
    df['high_rate_flag'] = (df['suku_bunga_euribor_3bln'] > mean_rate).astype(int)
    
    # Sector Grouping
    job_groups = mappings['job_sector_groups']
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
        else:
            return 'other'

    df['job_sector'] = df['pekerjaan'].apply(map_sector)
    
    # Category Mappings
    # We need to construct the mapping dictionary from the yaml structure
    # The yaml structure is flat for each category, which matches what map() expects
    
    # Apply all mappings
    for column, mapping in mappings.items():
        if column in df.columns and column != 'job_sector_groups':
             df[column] = df[column].map(mapping)
    
    # Save preprocessed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")
    return df

if __name__ == "__main__":
    config = load_config()
    input_path = config['paths']['raw_dataset']
    output_path = config['paths']['preprocessed_data']
    
    # Ensure paths are relative to project root if running from here
    if not os.path.exists(input_path) and os.path.exists(os.path.join('..', '..', input_path)):
         input_path = os.path.join('..', '..', input_path)
         output_path = os.path.join('..', '..', output_path)

    preprocess_data(input_path, output_path)
