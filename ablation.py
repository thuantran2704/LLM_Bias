import os
import pandas as pd
from ollama import Client
from tqdm import tqdm
import numpy as np

# ------------------------
# CONFIG
# ------------------------
OLLAMA_MODEL = "phi4-mini:3.8b"  # Change model name here
INCLUDE_JUSTIFICATION = True      # Set False to skip justification

TRANSCRIPT_FILE = "interview_transcripts_by_turkers.csv"
PROSODIC_FILE = "prosodic_features.csv"
FACIAL_DIR = "Facial_Features"
SMILE_DIR = "SmileData"

OUTPUT_DIR = "results" if INCLUDE_JUSTIFICATION else "results_no_justified"
os.makedirs(OUTPUT_DIR, exist_ok=True)

criteria = [
    "Overall", "RecommendHiring", "Colleague", "Engaged", "Excited",
    "EyeContact", "Smiled", "SpeakingRate", "NoFillers", "Friendly",
    "Paused", "EngagingTone", "StructuredAnswers", "Calm", "NotStressed",
    "Focused", "Authentic", "NotAwkward"
]

features = ["transcript", "prosodic", "facial", "smile"]

client = Client()

def load_transcripts():
    df = pd.read_csv(TRANSCRIPT_FILE, header=None, names=["id", "transcript"])
    print(f"Loaded {len(df)} transcripts.")
    return df

def load_prosodic():
    df = pd.read_csv(PROSODIC_FILE)
    print(f"Loaded {len(df)} prosodic rows, {df.shape[1]} features.")
    return df

def read_file(path, max_lines=20):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    return "".join(lines[:max_lines]).strip()

def summarize_prosodic(df_rows):
    numeric_cols = df_rows.select_dtypes(include=np.number).columns
    summary = []
    for col in numeric_cols:
        mean_val = df_rows[col].mean()
        summary.append(f"{col}_mean={mean_val:.3f}")
    return ", ".join(summary)

def get_candidate_features(idx, transcripts, prosodic):
    transcript = transcripts.iloc[idx]["transcript"]
    start = idx * 5
    end = start + 5
    prosodic_rows = prosodic.iloc[start:end]
    prosodic_summary = summarize_prosodic(prosodic_rows)
    facial_file = os.path.join(FACIAL_DIR, f"candidate{idx+1}.csv")
    facial_summary = read_file(facial_file, max_lines=10)
    smile_path = (
        os.path.join(SMILE_DIR, "pre", f"candidate{idx+1}.txt")
        if idx + 1 <= 69
        else os.path.join(SMILE_DIR, "post", f"candidate{idx+1 - 69}.txt")
    )
    smile_summary = read_file(smile_path, max_lines=10)
    return {
        "transcript": transcript,
        "prosodic": prosodic_summary,
        "facial": facial_summary,
        "smile": smile_summary
    }

def build_prompt(candidate_data, included_features, criterion):
    feature_text = "\n\n".join(
        f"{feat.capitalize()}:\n{candidate_data[feat]}" for feat in included_features
    )
    if INCLUDE_JUSTIFICATION:
        output_format = (
            "<score>;<short justification (1-2 sentences)>\n"
            "- Score must be integer 1-7\n"
            "- Do not write anything else"
        )
    else:
        output_format = (
            "<score>\n"
            "- Score must be integer 1-7\n"
            "- Do not write anything else"
        )

    return f"""You are an expert interviewer evaluator.

Rate the candidate for ONE criterion only: {criterion}.

Input data (features provided):
{feature_text}

Output format:
{output_format}
"""

def query_phi4(prompt):
    try:
        response_text = ""
        for chunk in client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"temperature": 0.3},
            stream=True
        ):
            if isinstance(chunk, dict) and "response" in chunk:
                response_text += chunk["response"]

        first_line = response_text.strip().split("\n")[0]
        if ";" in first_line:
            score_text = first_line.split(";")[0].strip()
        else:
            score_text = first_line.strip()

        try:
            score = int(score_text)
            if 1 <= score <= 7:
                return score, first_line
            else:
                print(f"Score out of range: {score_text}")
                print(f"Model output: {first_line}")
                return None, first_line
        except:
            print(f"Could not parse score: {score_text}")
            print(f"Model output: {first_line}")
            return None, first_line
    except Exception as e:
        print(f"Model query failed: {e}")
        return None, "Error"

def sanity_check():
    print("Running model sanity check...")
    if INCLUDE_JUSTIFICATION:
        test_prompt = """You are an expert interviewer evaluator.

Rate the candidate for ONE criterion only: Overall.

Input data: Candidate answered clearly and confidently.

Output format:
<score>;<short justification>"""
    else:
        test_prompt = """You are an expert interviewer evaluator.

Rate the candidate for ONE criterion only: Overall.

Input data: Candidate answered clearly and confidently.

Output format:
<score>"""

    response_text = ""
    for chunk in client.generate(
        model=OLLAMA_MODEL,
        prompt=test_prompt,
        options={"temperature": 0.3},
        stream=True
    ):
        if isinstance(chunk, dict) and "response" in chunk:
            response_text += chunk["response"]
    print("Sanity check response:")
    print(response_text.strip())
    print("End of sanity check\n")

def evaluate_candidates(start_idx, end_idx):
    transcripts = load_transcripts()
    prosodic = load_prosodic()
    candidate_ids = list(range(start_idx, end_idx + 1))

    for phase in ["all_features"] + [f"ablation_{f}" for f in features]:
        output_file = (
            os.path.join(OUTPUT_DIR, "All_features.csv")
            if phase == "all_features"
            else os.path.join(OUTPUT_DIR, f"{phase}.csv")
        )
        print(f"\n=== Starting phase: {phase} ===")
        rows = []

        for idx in tqdm(candidate_ids):
            candidate_data = get_candidate_features(idx - 1, transcripts, prosodic)
            included = features if phase == "all_features" else [
                f for f in features if f != phase.replace("ablation_", "")
            ]

            row = {"Participant": f"p{idx}", "Transcript": candidate_data["transcript"]}
            total = 0
            for crit in criteria:
                prompt = build_prompt(candidate_data, included, crit)
                score, output_line = query_phi4(prompt)
                if score is None:
                    score = 1
                row[crit] = score
                total += score
            row["Total"] = total
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df[["Participant", "Transcript"] + criteria + ["Total"]]
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Saved results to {output_file}")

    print("Finished all candidates.")

if __name__ == "__main__":
    print("Enter candidate range (e.g., 1 50):")
    start, end = map(int, input().split())
    sanity_check()
    evaluate_candidates(start, end)
