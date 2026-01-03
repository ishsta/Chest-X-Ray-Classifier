"""Debug script to test data loading"""
import pandas as pd
import pickle
from pathlib import Path

# Load config
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print("Config loaded successfully")
print(f"Images dir: {config['data']['images_dir']}")
print(f"CSV: {config['data']['filtered_csv']}")
print(f"Splits: {config['data']['split_file']}")

# Load CSV
csv_path = config['data']['filtered_csv']
if Path(csv_path).exists():
    df = pd.read_csv(csv_path)
    print(f"\nLoaded CSV: {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
else:
    print(f"\n⚠️ CSV not found at {csv_path}")
    
# Load splits
splits_path = config['data']['split_file']
if Path(splits_path).exists():
    with open(splits_path, 'rb') as f:
        splits = pickle.load(f)
    print(f"\nLoaded splits:")
    print(f"Train indices: {len(splits['train_indices'])}")
    print(f"Val indices: {len(splits['val_indices'])}")
    print(f"Test indices: {len(splits['test_indices'])}")
    
    # Check if indices are valid for the dataframe
    print(f"\nChecking indices against dataframe...")
    print(f"Max train index: {max(splits['train_indices'])}")
    print(f"Max val index: {max(splits['val_indices'])}")
    print(f"Max test index: {max(splits['test_indices'])}")
    print(f"Dataframe length: {len(df)}")
    print(f"Dataframe max index: {df.index.max()}")
    
    # Try to access the indices
    try:
        train_df = df.iloc[splits['train_indices']]
        print(f"\n✅ Successfully accessed train_df: {len(train_df)} rows")
    except Exception as e:
        print(f"\n❌ Error accessing train_df: {e}")
        print(f"   Trying alternative access...")
        # The problem is likely that the splits indices don't match the filtered CSV
        print(f"   Split indices min: {min(splits['train_indices'])}")
        print(f"   Split indices max: {max(splits['train_indices'])}")
else:
    print(f"\n⚠️ Splits not found at {splits_path}")
