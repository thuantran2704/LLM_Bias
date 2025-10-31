import os
import pandas as pd
from ollama import Client
from tqdm import tqdm
import time

# ------------------------
# CONFIG
# ------------------------
TRANSCRIPT_FILE = "interview_transcripts_by_turkers.csv"
PROSODIC_FILE = "prosodic_features.csv"
FACIAL_DIR = "Facial_Features"
SMILE_DIR = "SmileData"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

features = ["transcript", "prosodic", "facial", "smile"]

client = Client()

# ------------------------
# Load data
# ------------------------
def load_transcripts():
    df = pd.read_csv(TRANSCRIPT_FILE, header=None, names=["id", "transcript"])
    print(f"Loaded {len(df)} transcripts.")
    return df

def load_prosodic():
    df = pd.read_csv(PROSODIC_FILE)
    print(f"Loaded {len(df)} prosodic rows, {df.shape[1]} features.")
    return df

# ------------------------
# Read text helper
# ------------------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

# ------------------------
# Get candidate feature bundle
# ------------------------
def get_candidate_features(idx, transcripts, prosodic):
    transcript = transcripts.iloc[idx]["transcript"]
    start = idx * 5
    end = start + 5
    prosodic_rows = prosodic.iloc[start:end]
    prosodic_summary = prosodic_rows.to_string(index=False)

    facial = read_file(os.path.join(FACIAL_DIR, f"candidate{idx+1}.csv"))

    smile_path = (
        os.path.join(SMILE_DIR, "pre", f"candidate{idx+1}.txt")
        if idx + 1 <= 69
        else os.path.join(SMILE_DIR, "post", f"candidate{idx+1 - 69}.txt")
    )
    smile = read_file(smile_path)

    return {
        "transcript": transcript,
        "prosodic": prosodic_summary,
        "facial": facial,
        "smile": smile
    }

# ------------------------
# Build prompt
# ------------------------
def build_prompt_all_criteria(included_features, candidate_data):
    included_text = "\n\n".join(
        f"{feat.capitalize()}:\n{candidate_data[feat]}" for feat in included_features
    )
    crit_list = ", ".join(criteria)
    return f"""You are an expert interviewer evaluator.

Rate this candidate for the following criteria on a 1–7 scale:
{crit_list}

For each criterion, output EXACTLY one line in this format:
<CriterionName>: <score (1-7)>, <one-line justification>

Input data (features):
{included_text}

Output:
"""

# ------------------------
# Query model
# ------------------------
def query_phi4(prompt):
    try:
        response = client.generate(model="phi4-mini:3.8b", prompt=prompt, options={"temperature": 0.2})
        text = response.response.strip() if hasattr(response, "response") else str(response).strip()
        lines = [l.strip() for l in text.split("\n") if ":" in l]
        parsed = {}
        for line in lines:
            try:
                name, rest = line.split(":", 1)
                name = name.strip()
                score_part = ''.join(c for c in rest if c.isdigit())
                score = int(score_part[0]) if score_part else 1
                if name in criteria:
                    parsed[name] = score
            except:
                continue
        # Fill missing criteria with default 1
        for crit in criteria:
            if crit not in parsed:
                parsed[crit] = 1
        return parsed
    except Exception as e:
        print(f"Error: {e}")
        return {crit: 1 for crit in criteria}

# ------------------------
# Evaluate candidates with justifications study
# ------------------------
def evaluate_candidates(start_idx, end_idx):
    transcripts = load_transcripts()
    prosodic = load_prosodic()

    candidate_ids = list(range(start_idx, end_idx + 1))

    for phase in ["all_features"] + [f"ablation_{f}" for f in features]:
        output_file = os.path.join(OUTPUT_DIR, f"evaluation_{phase}.csv")

        all_results = []
        print(f"\n=== Starting phase: {phase} ===")

        for idx in tqdm(candidate_ids):
            candidate_data = get_candidate_features(idx-1, transcripts, prosodic)

            if phase == "all_features":
                included = features
            else:
                ablate_feat = phase.replace("ablation_", "")
                included = [f for f in features if f != ablate_feat]

            # --- 1. Query scores for CSV ---
            prompt_scores = build_prompt_all_criteria(included, candidate_data)
            scores = query_phi4(prompt_scores)

            # --- 2. Query justifications for study (not saved) ---
            prompt_justification = prompt_scores + "\n\nNow, for each criterion, provide a 1-2 sentence justification of your score."
            try:
                justification_response = client.generate(
                    model="phi4-mini:3.8b",
                    prompt=prompt_justification,
                    options={"temperature": 0.2}
                )
                justifications_text = justification_response.response.strip() if hasattr(justification_response, "response") else str(justification_response).strip()
                # Optional: print first 300 chars
                print(f"\n--- Candidate p{idx}, Phase {phase} Justifications ---")
                print(justifications_text[:300], "...\n")
            except Exception as e:
                print(f"Justification query error for Candidate p{idx}: {e}")

            # --- Build row in fixed column order for saving ---
            row = {"Participant": f"p{idx}", "Transcript": candidate_data["transcript"]}
            for crit in criteria:
                row[crit] = scores.get(crit, 1)
            all_results.append(row)

            time.sleep(0.2)

        # Save CSV with fixed order
        df_wide = pd.DataFrame(all_results)
        df_wide = df_wide[["Participant", "Transcript"] + criteria]
        df_wide.to_csv(output_file, index=False)
        print(f"✅ Saved results to {output_file}")

    print("\n✅ Finished all ablation phases.")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    evaluate_candidates(start, end)
