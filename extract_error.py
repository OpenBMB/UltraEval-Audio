import pandas as pd
import json

try:
    file_path = "res/qwen3-omni-speech/hsk1/res.xlsx"
    df = pd.read_excel(file_path)

    # Filter for match == 0
    errors = df[df["match"] == 0]

    if not errors.empty:
        # Get the first error
        row = errors.iloc[0]
        question_text = row["text"]
        inference_json = row["inference"]

        try:
            inference_data = json.loads(inference_json)
            model_reply = inference_data.get("text", "")
        except:
            model_reply = str(inference_json)

        print(f"FOUND_ERROR_SAMPLE")
        print(f"QUESTION: {question_text}")
        print(f"MODEL_REPLY: {model_reply}")
    else:
        print("No errors found.")

except Exception as e:
    print(f"Error: {e}")
