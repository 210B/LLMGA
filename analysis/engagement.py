import pandas as pd
import glob
import os
import re

# === Directories ===
data_dir = "experiment_data"
output_dir = "processed_output"
os.makedirs(output_dir, exist_ok=True)

# === Gather CSV files ===
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# === Normalize SessionID to format P00 ===
def normalize_session_id(session_id):
    match = re.search(r'\d+', str(session_id))
    if match:
        return f"P{int(match.group()):02d}"
    return session_id  # fallback

# === Process and Compute Metrics ===
all_data = []
engagement_ratio_summary = []

for file_path in csv_files:
    df = pd.read_csv(file_path, sep=';')

    # Normalize SessionID
    df["SessionID"] = df["SessionID"].apply(normalize_session_id)

    # Calculate AIResponseLength safely
    df["AIResponseLength"] = df["AIResponseText"].astype(str).str.len().replace(0, 1)

    # Compute EngagementRatio: Player vs LLM length
    df["EngagementRatio"] = df["ResponseLength"] / df["AIResponseLength"]

    # Extract metadata from filename
    filename = os.path.basename(file_path)
    label_type = "A" if "prompt" in filename else "B"
    label_source = "H" if "Hamlet" in filename else "M"

    # Add metadata columns
    df["LabelType"] = label_type
    df["LabelSource"] = label_source
    df["SourceFile"] = filename

    # Save processed individual file
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)

    # Collect for aggregation
    all_data.append(df)

    # Compute per-file average engagement ratio
    file_avg_ratio = df["EngagementRatio"].mean()
    engagement_ratio_summary.append({
        "SourceFile": filename,
        "LabelType": label_type,
        "LabelSource": label_source,
        "AverageEngagementRatio": file_avg_ratio
    })

    print(f"✅ Processed: {filename}")

# === Combine and Save All Data ===
combined_df = pd.concat(all_data, ignore_index=True)
combined_df.to_csv(os.path.join(output_dir, "all_data_combined.csv"), index=False)

# === Save Summary Per File ===
summary_df = pd.DataFrame(engagement_ratio_summary)
summary_df.to_csv(os.path.join(output_dir, "engagement_ratio_summary.csv"), index=False)

# === Compute and Save Group-Level Engagement Ratio Averages (A/B) ===
group_ratio_avg = combined_df.groupby("LabelType")["EngagementRatio"].mean().reset_index()
group_ratio_avg.columns = ["LabelType", "AverageEngagementRatio"]
group_ratio_avg.to_csv(os.path.join(output_dir, "group_engagement_ratio_avg.csv"), index=False)

print("\n📊 Group-wise average engagement ratio:")
print(group_ratio_avg)

print("✅ All processing complete.")

