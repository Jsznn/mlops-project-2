import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    # Get the project root directory (assuming this script is run from root or src/...)
    # Adjust path resolution to be robust
    
    # If running from root
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    # If running from src/ or tests/ (one level down)
    parent_path = os.path.join("..", config_path)
    if os.path.exists(parent_path):
         with open(parent_path, 'r') as f:
            return yaml.safe_load(f)
            
    # If running from src/api/ (two levels down)
    grandparent_path = os.path.join("..", "..", config_path)
    if os.path.exists(grandparent_path):
         with open(grandparent_path, 'r') as f:
            return yaml.safe_load(f)

    raise FileNotFoundError(f"Configuration file not found at {config_path} or parent directories")

def load_mappings(mapping_path: str = "config/mappings.yaml") -> Dict[str, Any]:
    """Load mappings from a YAML file."""
    return load_config(mapping_path)
