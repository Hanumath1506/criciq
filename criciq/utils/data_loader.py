import pandas as pd
import os

def  load_matches(folder_path):
    all_files = []

    for file in os.listdir(folder_path):
        if file.endswith('.csv') and '_info' not in file:
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            all_files.append(df)

    combined = pd.concat(all_files, ignore_index=True)
    return combined