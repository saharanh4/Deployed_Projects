import pandas as pd
import os

FILE_PATH = ""

if "_target_cache" not in globals():
    _target_cache = {'path': None, 'target': None, 'task': None}

def load_data(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, sep=None, engine='python')

def ask_target(df):
    print("Columns:",list(df.columns))
    while True:
        target = input("Enter the target column").strip()
        if target in df.columns:
            return target
        print(f"'{target}' not found. Try again.")

def get_target(df, file_path):
    if _target_cache['path'] == file_path and _target_cache['target'] in df.columns:
        print(f"Same file, reusing target: {_target_cache['target']}")
        return _target_cache['target']
    target = ask_target(df)
    _target_cache.update({'path': file_path, 'target': target, 'task': None})
    return  target

df = load_data(FILE_PATH)
TARGET = get_target(df, FILE_PATH)