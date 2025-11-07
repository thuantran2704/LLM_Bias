import csv, os, re
from ollama import Client
from tqdm import tqdm
import pandas as pd

# ------------------------
# CONFIG
# ------------------------
MODEL = "phi4-mini:3.8b"
INPUT = "interview_transcripts_by_turkers.csv"
FACIAL_FEATURES_DIR = "Facial_Features"
SMILE_DATA_DIR = "SmileData"
PROSODIC_FILE = "prosodic_features.csv"
PRE_COUNT = 69  # number of pre candidates

CRITERIA = [
    "Overall","RecommendHiring","Colleague","Engaged","Excited","EyeContact",
    "Smiled","SpeakingRate","NoFillers","Friendly","Paused","EngagingTone",
    "StructuredAnswers","Calm","NotStressed","Focused","Authentic","NotAwkward"
]

# ------------------------
# CLIENT
# ------------------------
client = Client()

# ------------------------
# LOADERS
# ------------------------
def load_prosodic_features(participant_index, file_path=PROSODIC_FILE):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2:
                return ""
            header = lines[0]
            start = 1 + participant_index * 5
            end = start + 5
            if start >= len(lines):
                return ""
            candidate_lines = lines[start:end]
            return header + "\n" + "\n".join(candidate_lines)
    except:
        return ""

def load_facial_features(participant_index):
    files = sorted(os.listdir(FACIAL_FEATURES_DIR))
    if participant_index < len(files):
        path = os.path.join(FACIAL_FEATURES_DIR, files[participant_index])
        try:
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        except:
            return ""
    return ""

def load_smile_data(participant_index):
    if participant_index < PRE_COUNT:
        folder = "pre"
        file_index = participant_index
    else:
        folder = "post"
        file_index = participant_index - PRE_COUNT

    folder_path = os.path.join(SMILE_DATA_DIR, folder)
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        if file_index < len(files):
            path = os.path.join(folder_path, files[file_index])
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except:
        pass
    return ""

# ------------------------
# PROMPT + SCORING
# ------------------------
def build_prompt(features_dict, criterion):
    feature_lines = []
    for key, value in features_dict.items():
        feature_lines.append(f"{key}:\n\"\"\"{value}\"\"\"")
    features_text = "\n\n".join(feature_lines)
    
    prompt = f"""
You are an interview evaluator.
Rate the candidate on the criterion "{criterion}" based on the following information:

{features_text}

Rules:
- Output exactly in this format:
  score;justification
- score = integer 1–7
- justification = 1–2 sentences explaining the score
- Do NOT include extra text
"""
    return prompt.strip()

def ask_score_dynamic(features_dict, criterion, max_retries=5):
    for _ in range(max_retries):
        prompt = build_prompt(features_dict, criterion)
        try:
            r = client.generate(model=MODEL, prompt=prompt)
            text = r["response"].strip()
            if ";" in text:
                score_str = text.split(";", 1)[0].strip()
                if score_str.isdigit() and 1 <= int(score_str) <= 7:
                    return int(score_str)
        except:
            pass
    return 0

def grade_candidate(features_dict):
    scores = {}
    for c in CRITERIA:
        scores[c] = ask_score_dynamic(features_dict, c)
    scores["Total"] = sum(scores[c] for c in CRITERIA)
    return scores

# ------------------------
# CORE GRADING FUNCTION
# ------------------------
def run_grading(output_file, features_to_include, transcripts, s):
    rows = []
    for i, transcript in enumerate(tqdm(transcripts, desc=f"Grading {output_file}")):
        # dynamically load selected features
        all_features = {
            "Transcript": transcript,
            "Facial_Features": load_facial_features(s + i),
            "SmileData": load_smile_data(s + i),
            "Prosodic_Features": load_prosodic_features(s + i)
        }
        # filter out removed features
        features = {k: v for k, v in all_features.items() if k in features_to_include}
        scores = grade_candidate(features)
        row = {"Participant": s + i + 1}
        row.update(scores)
        rows.append(row)

    header = ["Participant"] + CRITERIA + ["Total"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved to {output_file}")

# ------------------------
# MAIN EXECUTION
# ------------------------
with open(INPUT, encoding="utf-8") as f:
    transcripts = [r[0] for r in csv.reader(f) if r]

s, e = map(int, input(f"Range (0–{len(transcripts)-1}) start,end: ").split(","))
subset = transcripts[s:e]

# All features
all_features = ["Transcript", "Facial_Features", "SmileData", "Prosodic_Features"]
run_grading("all_feature.csv", all_features, subset, s)

# Ablation (remove one feature at a time)
for ftr in all_features:
    features_left = [x for x in all_features if x != ftr]
    output_name = f"ablation_{ftr}.csv"
    run_grading(output_name, features_left, subset, s)
