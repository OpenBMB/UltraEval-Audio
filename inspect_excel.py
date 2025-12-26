import pandas as pd

try:
    file_path = "res/qwen3-omni-speech/hsk1/res.xlsx"
    df = pd.read_excel(file_path)

    print("File Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())

    # Check for some basic stats if columns are numeric
    print("\nDescribe:")
    print(df.describe().to_string())

except Exception as e:
    print(f"Error reading excel file: {e}")
