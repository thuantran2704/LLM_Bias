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
# Build prompt for one criterion
# ------------------------
def build_prompt(candidate_data, included_features, criterion):
    feature_text = "\n\n".join(f"{feat.capitalize()}:\n{candidate_data[feat]}" for feat in included_features)
    return f"""You are an expert interviewer evaluator.

Rate the candidate for ONE criterion only: {criterion}.

Required output format:
<score>;<short justification (1-2 sentences)>
- Score must be an integer from 1 to 7.
- Do not include anything else besides <score>;<justification>.

Input data:
{feature_text}

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

        # Take first line and parse
        first_line = response_text.strip().split("\n")[0]
        if ";" in first_line:
            score_text = first_line.split(";")[0].strip()
            try:
                score = int(score_text)
                if 1 <= score <= 7:
                    return score, first_line
                else:
                    print(f"⚠️ Score out of range: {score_text}")
                    print(f"❗ Model output: {first_line}")
                    return None, first_line
            except:
                print(f"⚠️ Could not parse score: {score_text}")
                print(f"❗ Model output: {first_line}")
                return None, first_line
        else:
            print(f"⚠️ No semicolon in output.")
            print(f"❗ Model output: {first_line}")
            return None, first_line
    except Exception as e:
        print(f"❌ Model query failed: {e}")
        return None, "Error"


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
        else:
            feature_name = phase.replace("ablation_", "")
            output_file = os.path.join(OUTPUT_DIR, f"ablation_{feature_name}.csv")

        print(f"\n=== Starting phase: {phase} ===")
        rows = []

        for idx in tqdm(candidate_ids):
            candidate_data = get_candidate_features(idx-1, transcripts, prosodic)

            if phase == "all_features":
                included = features
            else:
                ablate_feat = phase.replace("ablation_", "")
                included = [f for f in features if f != ablate_feat]

            row = {"Participant": f"p{idx}", "Transcript": candidate_data["transcript"]}
            total = 0

            for crit in criteria:
                prompt = build_prompt(candidate_data, included, crit)
                score, _ = query_phi4(prompt)
                if score is None:
                    score = 1  # fallback if model fails
                row[crit] = score
                total += score

            row["Total"] = total
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[["Participant", "Transcript"] + criteria + ["Total"]]
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"✅ Saved results to {output_file}")

    print("\n✅ Finished all candidates.")

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    evaluate_candidates(start, end)
