import os
import pandas as pd
import glob

# ------------------------
# CONFIG
# ------------------------
ALL_FEATURES_FILE = "results/evaluation_all_features.csv"  # reference
ABLATION_DIR = "results"
OUTPUT_FILE = "results/squared_errors.csv"

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

# ------------------------
# Load all-features reference
# ------------------------
df_ref = pd.read_csv(ALL_FEATURES_FILE)

# ------------------------
# Find all ablation files
# ------------------------
ablation_files = glob.glob(os.path.join(ABLATION_DIR, "evaluation_ablation_*.csv"))
ablation_files.sort()  # optional, ensures consistent order

all_errors = []

for file in ablation_files:
    phase_name = os.path.basename(file).replace("evaluation_", "").replace(".csv", "")
    df_ab = pd.read_csv(file)

    # Ensure rows are aligned by Participant
    df_ab = df_ab.set_index("Participant").reindex(df_ref["Participant"]).reset_index()

    # Compute squared error for each criterion
    error_rows = []
    for idx, row_ref in df_ref.iterrows():
        row_ab = df_ab.iloc[idx]
        error_row = {"Participant": row_ref["Participant"], "Phase": phase_name}
        for crit in criteria:
            error_row[crit] = (row_ab[crit] - row_ref[crit]) ** 2
        error_rows.append(error_row)

    all_errors.extend(error_rows)

# ------------------------
# Save results
# ------------------------
df_errors = pd.DataFrame(all_errors)
df_errors = df_errors[["Participant", "Phase"] + criteria]
df_errors.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Squared error file saved to {OUTPUT_FILE}")
