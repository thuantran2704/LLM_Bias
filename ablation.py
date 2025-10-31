import os
import pandas as pd
from ollama import Client
from tqdm import tqdm
import time

client = Client()

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

features = ["transcript", "prosodic", "facial", "smile_pre", "smile_post"]

# ------------------------
# Load data
# ------------------------
def load_data():
    transcripts = pd.read_csv("interview_transcripts_by_turkers.csv")
    prosodic = pd.read_csv("prosodic_features.csv")
    return transcripts, prosodic

# ------------------------
# Helper to read text files
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
    transcript = transcripts.iloc[idx]["Transcript"] if "Transcript" in transcripts.columns else ""
    prosodic_rows = prosodic.iloc[idx*5:(idx+1)*5]
    prosodic_summary = prosodic_rows.to_string(index=False)

    facial = read_file(f"Facial_Features/candidate{idx+1}.csv")
    smile_pre = read_file(f"SmileData/pre/candidate{idx+1}.txt")
    smile_post = read_file(f"SmileData/post/candidate{idx+1}.txt")

    return {
        "transcript": transcript,
        "prosodic": prosodic_summary,
        "facial": facial,
        "smile_pre": smile_pre,
        "smile_post": smile_post
    }

# ------------------------
# Build one big prompt for all 18 criteria
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

Example:
Overall: 6, Confident and friendly
RecommendHiring: 5, Minor hesitation but professional

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
        parsed = []
        for line in lines:
            try:
                name, rest = line.split(":", 1)
                name = name.strip()
                score_part = ''.join(c for c in rest if c.isdigit())
                score = int(score_part[0]) if score_part else 1
                justification = rest.split(",", 1)[-1].strip() if "," in rest else "No justification"
                if name in criteria:
                    parsed.append((name, score, justification))
            except:
                continue
        return parsed
    except Exception as e:
        print(f"Error: {e}")
        return [(c, 1, "Error") for c in criteria]

# ------------------------
# Evaluate range of candidates
# ------------------------
def evaluate_candidates(start_idx, end_idx, output_file="evaluation_results.csv"):
    transcripts, prosodic = load_data()

    # Resume support
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        done = set(zip(existing.Candidate, existing.Phase))
    else:
        done = set()

    for idx in tqdm(range(start_idx-1, end_idx)):
        candidate_data = get_candidate_features(idx, transcripts, prosodic)
        candidate_id = idx + 1

        for phase in ["all_features"] + [f"ablation_{f}" for f in features]:
            if (candidate_id, phase) in done:
                continue

            if phase == "all_features":
                included = features
            else:
                ablate_feat = phase.replace("ablation_", "")
                included = [f for f in features if f != ablate_feat]

            prompt = build_prompt_all_criteria(included, candidate_data)
            parsed = query_phi4(prompt)

            results = []
            for crit, score, justification in parsed:
                results.append({
                    "Candidate": candidate_id,
                    "Phase": phase,
                    "Criterion": crit,
                    "Score": score,
                    "Justification": justification
                })

            df = pd.DataFrame(results)
            df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
            time.sleep(0.2)

    print(f"✅ Finished. Results saved to {output_file}")


# ------------------------
# Run script
# ------------------------
if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    evaluate_candidates(start, end)
