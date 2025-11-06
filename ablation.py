import csv, os, itertools
from ollama import Client
from tqdm import tqdm
import pandas as pd

MODEL = "phi4-mini:3.8b"
INPUT = "interview_transcripts_by_turkers.csv"
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
def load_prosodic_features(idx, file_path="prosodic_features.csv"):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            if len(lines) < 2:
                return ""
            header = lines[0]
            start = 1 + idx * 5
            end = start + 5
            if start >= len(lines):
                return ""
            return header + "\n" + "\n".join(lines[start:end])
    except:
        return ""

def load_facial_features(idx):
    files = sorted(os.listdir(FACIAL_FEATURES_DIR))
    if idx < len(files):
        path = os.path.join(FACIAL_FEATURES_DIR, files[idx])
        try:
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        except:
            return ""
    return ""

def load_smile_data(idx):
    folder = "pre" if idx < PRE_COUNT else "post"
    file_index = idx if idx < PRE_COUNT else idx - PRE_COUNT
    folder_path = os.path.join(SMILE_DATA_DIR, folder)
    try:
        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])
        if file_index < len(files):
            with open(os.path.join(folder_path, files[file_index]), "r", encoding="utf-8") as f:
                return f.read()
    except:
        pass
    return ""

# ------------------------
# Prompt & scoring
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
        except:
            pass
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
# Dynamic ablation
# ------------------------
# base feature dict for all candidates
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

# Get all feature names (keys) except Transcript (usually keep it)
all_features = list(base_features_list[0].keys())
all_features.remove("Transcript")  # optional: never ablate transcript

# Generate ablation combinations
ablation_sets = {"full": all_features.copy()}
for r in range(1, len(all_features)+1):
    for combo in itertools.combinations(all_features, r):
        name = "ablation_" + "_".join([f for f in all_features if f not in combo])
        ablation_sets[name] = list(combo)

# ------------------------
# Run grading for each ablation set
# ------------------------
for ablation_name, features_to_include in ablation_sets.items():
    print(f"\nGrading candidates: {ablation_name}")
    rows = []
    
    for i, base_features in enumerate(tqdm(base_features_list, desc=f"Grading {ablation_name}")):
        features = {"Transcript": base_features["Transcript"]}
        for f in features_to_include:
            features[f] = base_features[f]
        scores = grade_candidate(features)
        row = {"Participant": s + i + 1, "Transcript": ""}
        row.update(scores)
        rows.append(row)
    
    output_file = f"candidate_grades_{ablation_name}.csv"
    header = ["Participant","Transcript"] + CRITERIA + ["Total"]
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Saved to {output_file}")
