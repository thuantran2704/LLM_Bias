import csv, os
from ollama import Client
from tqdm import tqdm
import pandas as pd

MODEL = "phi4-mini:3.8b"
INPUT = "interview_transcripts_by_turkers.csv"
OUTPUT = "candidate_grades.csv"
FACIAL_FEATURES_DIR = "Facial_Features"
SMILE_DATA_DIR = "SmileData"
PRE_COUNT = 69

CRITERIA = [
    "Overall","RecommendHiring","Colleague","Engaged","Excited","EyeContact",
    "Smiled","SpeakingRate","NoFillers","Friendly","Paused","EngagingTone",
    "StructuredAnswers","Calm","NotStressed","Focused","Authentic","NotAwkward"
]

client = Client()

# ------------------------
# Feature loaders
# ------------------------
def load_prosodic_features(participant_index, file_path="prosodic_features.csv"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2: return ""
            header = lines[0]
            start = 1 + participant_index * 5
            end = start + 5
            if start >= len(lines): return ""
            return header + "\n" + "\n".join(lines[start:end])
    except: return ""

def load_facial_features(participant_index):
    files = sorted(os.listdir(FACIAL_FEATURES_DIR))
    if participant_index < len(files):
        path = os.path.join(FACIAL_FEATURES_DIR, files[participant_index])
        try:
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        except: return ""
    return ""

def load_smile_data(participant_index):
    folder = "pre" if participant_index < PRE_COUNT else "post"
    file_index = participant_index if participant_index < PRE_COUNT else participant_index - PRE_COUNT
    folder_path = os.path.join(SMILE_DATA_DIR, folder)
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        if file_index < len(files):
            with open(os.path.join(folder_path, files[file_index]), "r", encoding="utf-8") as f:
                return f.read()
    except: pass
    return ""

# ------------------------
# Prompt builder & scoring
# ------------------------
def build_prompt(features_dict, criterion):
    feature_lines = [f"{k}:\n\"\"\"{v}\"\"\"" for k, v in features_dict.items()]
    features_text = "\n\n".join(feature_lines)
    return f"""
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
        except: pass
    return 0

def grade_candidate(features_dict):
    scores = {c: ask_score_dynamic(features_dict, c) for c in CRITERIA}
    scores["Total"] = sum(scores.values())
    return scores

# ------------------------
# Load transcripts
# ------------------------
with open(INPUT, encoding="utf-8") as f:
    transcripts = [r[0] for r in csv.reader(f) if r]

s, e = map(int, input(f"Range (0–{len(transcripts)-1}) start,end: ").split(","))
subset = transcripts[s:e]

# ------------------------
# Build base features dict for all candidates
# ------------------------
base_features_list = []
for i, transcript in enumerate(subset):
    idx = s + i
    base_features = {
        "Transcript": transcript,
        "Facial_Features": load_facial_features(idx),
        "Prosodic_Features": load_prosodic_features(idx),
        "SmileData": load_smile_data(idx)
    }
    base_features_list.append(base_features)

# ------------------------
# Determine ablation sets dynamically (Transcript can be ablated now)
# ------------------------
all_features = list(base_features_list[0].keys())  # include Transcript

ablation_sets = {"full": all_features.copy()}  # full run with all features
# single-feature ablations
for f in all_features:
    ablation_sets[f"ablation_{f}"] = [feat for feat in all_features if feat != f]

# ------------------------
# Grade candidates for each ablation set
# ------------------------
for ablation_name, features_to_include in ablation_sets.items():
    print(f"\nGrading candidates: {ablation_name}")
    rows = []
    for i, base_features in enumerate(tqdm(base_features_list, desc=f"Grading {ablation_name}")):
        features = {}
        for f in features_to_include:
            features[f] = base_features[f]
        scores = grade_candidate(features)
        row = {"Participant": s + i + 1, "Transcript": ""}
        row.update(scores)
        rows.append(row)
    
    output_file = OUTPUT.replace(".csv", f"_{ablation_name}.csv")
    file_exists = os.path.exists(output_file)

    with open(output_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()  # write header only if file is new
        writer.writerows(rows)
    
    print(f"✅ Saved to {output_file}")
