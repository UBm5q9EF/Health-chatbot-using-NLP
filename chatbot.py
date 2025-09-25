import json
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz, process
import re

# ------------------ Setup ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load diseases from JSON
with open("diseases.json", "r", encoding="utf-8") as f:
    data = json.load(f)
DISEASES = data.get("diseases", [])

# Extract known symptoms
COMMON_SYMPTOMS = list({sym.lower() for d in DISEASES for sym in d.get("symptoms", [])})

# Synonyms mapping
SYMPTOM_SYNONYMS = {
    "body ache": "muscle pain",
    "body aches": "muscle pain",
    "tired": "fatigue",
    "weak": "fatigue",
    "lightheaded": "dizziness",
    "can't breathe": "shortness of breath",
    "breathing difficulty": "shortness of breath",
    "watery eyes": "itchy eyes",
    "sweating a lot": "sweating",
    "low mood": "persistent low mood",
    "heart racing": "palpitations",
    "numbness in feet": "numbness",
    "pain while swallowing": "sore throat",
    "stomach hurts": "abdominal pain",
    "stomach pain": "abdominal pain",
    "vomiting": "nausea"
}

# ------------------ Helpers ------------------
def normalize(s: str) -> str:
    return s.lower().strip()

def map_to_known_symptom(text: str) -> str:
    text = normalize(text)
    if text in SYMPTOM_SYNONYMS:
        return SYMPTOM_SYNONYMS[text]
    best = process.extractOne(text, COMMON_SYMPTOMS, scorer=fuzz.partial_ratio)
    if best and best[1] >= 75:
        return best[0]
    return text

def extract_symptoms(text: str):
    tokens = re.split(r"[,\n]+", text)
    cleaned = [map_to_known_symptom(tok.strip()) for tok in tokens if tok.strip()]
    return list(set(cleaned))

def diagnose_from_symptoms(symptom_list, min_symptoms_required=2, top_k=3):
    if not symptom_list or len(symptom_list) < min_symptoms_required:
        return {"error": f"⚠️ Please provide at least {min_symptoms_required} symptoms."}

    results = []
    for d in DISEASES:
        disease_symptoms = [s.lower() for s in d.get("symptoms", [])]
        match_scores = []
        for us in symptom_list:
            best = process.extractOne(us, disease_symptoms, scorer=fuzz.token_sort_ratio)
            if best:
                match_scores.append(best[1])

        good_matches = sum(1 for sc in match_scores if sc >= 70)
        avg_score = sum(match_scores) / len(match_scores) if match_scores else 0
        composite = good_matches * 10 + avg_score / 10

        results.append({
            "name": d["name"],
            "type": d.get("type", ""),
            "causes": d.get("causes", ""),
            "treatment": d.get("treatment", ""),
            "remedies": d.get("remedies", []),
            "prevention": d.get("prevention", ""),
            "matched_symptom_count": good_matches,
            "avg_symptom_similarity": round(avg_score, 1),
            "composite_score": round(composite, 2),
            "symptoms": disease_symptoms
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return {"matches": results[:top_k]}

def remedies_for_disease(disease_name: str):
    disease_name = normalize(disease_name)
    for d in DISEASES:
        if normalize(d["name"]) == disease_name:
            return {
                "name": d["name"],
                "remedies": d.get("remedies", []),
                "treatment": d.get("treatment", ""),
                "prevention": d.get("prevention", ""),
                "causes": d.get("causes", "")
            }
    return {"message": "⚠️ Disease not found. Try again with a valid name."}

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/diagnose", methods=["POST"])
def api_diagnose():
    payload = request.json or {}
    mode = payload.get("mode", "predict")
    text = payload.get("input", "")

    if mode == "predict":
        extracted = extract_symptoms(text)
        result = diagnose_from_symptoms(extracted)
    elif mode == "remedies":
        result = remedies_for_disease(text)
    else:
        result = {"error": "Invalid mode. Use 'predict' or 'remedies'."}

    return jsonify(result)

# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
