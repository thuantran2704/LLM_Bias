import os
import pandas as pd
from ollama import Client
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
TRANSCRIPT_FILE = "interview_transcripts_by_turkers.csv"
PROSODIC_FILE = "prosodic_features.csv"
FACIAL_DIR = "Facial_Features"
SMILE_DIR = "SmileData"
OUTPUT_DIR = "results"
JUSTIFIED_DIR = "justified_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JUSTIFIED_DIR, exist_ok=True)

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
# Helpers
# ------------------------
def read_file(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

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
# Prompt builders
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
<CriterionName>: <score (1-7)>

Input data (features):
{included_text}

Output:
"""

def build_prompt_with_justification(included_features, candidate_data):
    included_text = "\n\n".join(
        f"{feat.capitalize()}:\n{candidate_data[feat]}" for feat in included_features
    )
    crit_list = ", ".join(criteria)
    return f"""You are an expert interviewer evaluator.

Rate this candidate for the following criteria (1–7 scale):
{crit_list}

For each criterion, output EXACTLY one line in this format:
<CriterionName>: <score (1-7)>, <short justification (1–2 sentences) explaining the rating>

Input data (features provided):
{included_text}

Output:
"""

# ------------------------
# Query model
# ------------------------
def query_phi4(prompt):
    try:
        response_text = ""
        for chunk in client.generate(
            model="phi4-mini:3.8b",
            prompt=prompt,
            options={"temperature": 0.2},
            stream=True
        ):
            if isinstance(chunk, dict) and "response" in chunk:
                response_text += chunk["response"]

        text = response_text.strip()
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

        for crit in criteria:
            if crit not in parsed:
                parsed[crit] = 1

        return parsed, text
    except Exception as e:
        print(f"Error: {e}")
        return {crit: 1 for crit in criteria}, "Error"

# ------------------------
# Main evaluation
# ------------------------
def evaluate_candidates(start_idx, end_idx):
    transcripts = load_transcripts()
    prosodic = load_prosodic()
    candidate_ids = list(range(start_idx, end_idx + 1))

    for phase in ["all_features"] + [f"ablation_{f}" for f in features]:
        if phase == "all_features":
            output_file = os.path.join(OUTPUT_DIR, "All_features.csv")
            justified_file = os.path.join(JUSTIFIED_DIR, "All_features_justified.csv")
        else:
            feature_name = phase.replace("ablation_", "")
            output_file = os.path.join(OUTPUT_DIR, f"ablation_{feature_name}.csv")
            justified_file = os.path.join(JUSTIFIED_DIR, f"ablation_{feature_name}_justified.csv")

        all_results = []
        justified_results = []
        print(f"\n=== Starting phase: {phase} ===")

        for idx in tqdm(candidate_ids):
            candidate_data = get_candidate_features(idx-1, transcripts, prosodic)

            if phase == "all_features":
                included = features
            else:
                ablate_feat = phase.replace("ablation_", "")
                included = [f for f in features if f != ablate_feat]

            # --- Attempt 1: Ablation scores ---
            prompt_scores = build_prompt_all_criteria(included, candidate_data)
            scores, _ = query_phi4(prompt_scores)

            row = {"Participant": f"p{idx}", "Transcript": candidate_data["transcript"]}
            for crit in criteria:
                row[crit] = scores.get(crit, 1)
            row["Total"] = sum(row[c] for c in criteria)
            all_results.append(row)

            # --- Attempt 2: Justification ---
            prompt_just = build_prompt_with_justification(included, candidate_data)
            _, justification_text = query_phi4(prompt_just)
            justified_results.append({
                "Participant": f"p{idx}",
                "Transcript": candidate_data["transcript"],
                "JustifiedOutput": justification_text
            })

        # Save both CSVs
        df = pd.DataFrame(all_results)
        df = df[["Participant", "Transcript"] + criteria + ["Total"]]
        df.to_csv(output_file, index=False, encoding="utf-8-sig")

        df_just = pd.DataFrame(justified_results)
        df_just.to_csv(justified_file, index=False, encoding="utf-8-sig")

        print(f"✅ Saved results to {output_file}")
        print(f"✅ Saved justifications to {justified_file}")

    print("\n✅ Finished all ablation and justification phases.")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    evaluate_candidates(start, end)
