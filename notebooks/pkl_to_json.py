import os
import pickle
import json
import sys

def is_json_serializable(data):
    try:
        json.dumps(data)
        return True
    except (TypeError, OverflowError):
        return False
    
def try_preprocess(data):
    result = {}
    for key, value in data.items():
        if isinstance(key, tuple):
            key = "_".join(map(str, key))
        result[key] = value
    return result

def process_file(filepath):
    if not filepath.endswith('.pkl'):
        return False
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
    except (pickle.PickleError, EOFError):
        return False
    
    if not isinstance(data, dict):
        return False
    
    data = try_preprocess(data)
    
    if not is_json_serializable(data):
        return False
    
    json_filepath = filepath[:-4] + '.json'
    try:
        with open(json_filepath, 'w') as f:
            json.dump(data, f, indent=2)
        os.remove(filepath)
        return True
    except (IOError, OSError):
        return False

def process_directory(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            process_file(filepath)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
    
    process_directory(directory)
    print("Processing complete.")