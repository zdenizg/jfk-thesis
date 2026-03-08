import os
import pandas as pd

# Folder containing your CSVs
folder_path = "/Users/furkandemir/Desktop/Thesis/organizers/results"

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

# Read and combine
df_list = []
for file in csv_files:
    file_path = os.path.join(folder_path, file)
    df = pd.read_csv(file_path)
    df["source_file"] = file  # optional: helps track origin
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# Save if you want
combined_df.to_csv("jfk_categorization_combined.csv", index=False)

print("Done! Combined shape:", combined_df.shape)
