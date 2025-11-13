import os
import yaml
import pandas as pd

def extract_yaml_to_csv(yaml_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(yaml_folder):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                with open(os.path.join(root, file), 'r') as f:
                    data = yaml.safe_load(f)
                    df = pd.DataFrame(data)
                    symbol = df['Symbol'].iloc[0]
                    csv_path = os.path.join(output_folder, f"{symbol}.csv")
                    df.to_csv(csv_path, index=False)
                    print(f"Saved {csv_path}")

if __name__ == "__main__":
    extract_yaml_to_csv("data/yaml_data", "data/csv_data")
