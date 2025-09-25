import json
from flask import Flask, request, jsonify, render_template
from fuzzywuzzy import fuzz, process
import spacy

# ------------------ Setup ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load diseases database
with open("diseases.json", "r", encoding="utf-8") as f:
    data = json.load(f)
DISEASES = data.get("diseases", [])

# ✅ Build dynamic list of all known symptoms from dataset
COMMON_SYMPTOMS = list(
    {sym.lower() for d in DISEASES for sym in d.get("symptoms", [])}
)


# ------------------ Helpers ------------------
def normalize(s: str) -> str:
    return s.lower().strip()


def extract_symptoms(text):
    """Extract symptoms from free text using spaCy + keyword/phrase matching."""
    doc = nlp(text)
    extracted = []

    # Token matching
    for token in doc:
        if token.text.lower() in COMMON_SYMPTOMS:
            extracted.append(token.text.lower())

    # Phrase (noun chunks like "chest pain")
    for chunk in doc.noun_chunks:
        if chunk.text.lower() in COMMON_SYMPTOMS:
            extracted.append(chunk.text.lower())

    return list(set(extracted))  # unique symptoms


def diagnose_from_symptoms(symptom_list, min_symptoms_required=3, top_k=3):
    """Match symptoms against diseases using fuzzy matching."""
    if not symptom_list or len(symptom_list) < min_symptoms_required:
        return {"error": f"⚠️ Please provide at least {min_symptoms_required} symptoms."}

    results = []
    for d in DISEASES:
        disease_symptoms = [s.lower() for s in d.get("symptoms", [])]

        match_scores = []
        for us in symptom_list:
            best = process.extractOne(us, disease_symptoms, scorer=fuzz.token_sort_ratio) or ("", 0)
            match_scores.append(best[1])

        good_matches = sum(1 for sc in match_scores if sc >= 70)
        avg_score = sum(match_scores) / len(match_scores) if match_scores else 0
        composite = good_matches * 10 + avg_score / 10

        results.append({
            "name": d["name"],
            "type": d["type"],
            "matched_symptom_count": good_matches,
            "avg_symptom_similarity": round(avg_score, 1),
            "composite_score": round(composite, 2),
            "symptoms": d.get("symptoms", []),
            "treatment": d.get("treatment", ""),
            "remedies": d.get("remedies", []),
            "prevention": d.get("prevention", "")
        })

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return {"matches": results[:top_k]}


def remedies_for_disease(disease_name: str):
    """Fetch remedies for a given disease name"""
    disease_name = normalize(disease_name)
    for d in DISEASES:
        if normalize(d["name"]) == disease_name:
            return {
                "name": d["name"],
                "remedies": d.get("remedies", []),
                "treatment": d.get("treatment", ""),
                "prevention": d.get("prevention", "")
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
        # ✅ First try NLP
        extracted = extract_symptoms(text)

        if extracted:
            # If NLP got at least 2 symptoms, use them
            min_required = 2 if len(extracted) >= 2 else 3
            parts = extracted
        else:
            # fallback: manual comma-separated input
            parts = [p.strip() for p in text.replace("\n", ",").split(",") if p.strip()]
            min_required = 3

        result = diagnose_from_symptoms(parts, min_symptoms_required=min_required, top_k=4)

    elif mode == "remedies":
        result = remedies_for_disease(text)

    else:
        result = {"error": "Invalid mode. Use 'predict' or 'remedies'."}

    return jsonify(result)


# ------------------ Run ------------------
if __name__ == "__main__":
    print("🚀 Health Chatbot running on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
