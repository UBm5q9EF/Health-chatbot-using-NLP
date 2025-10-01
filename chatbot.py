import os
import json
import spacy
from flask import Flask, request, jsonify, render_template
from flask_session import Session
from flask_cors import CORS
from fuzzywuzzy import process, fuzz
from datetime import datetime

# ------------------ Flask Setup ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "super-secret-key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app)

# ------------------ Load spaCy Trained Model ------------------
try:
    nlp = spacy.load("symptom_ner_model")
except Exception as e:
    raise RuntimeError("❌ Failed to load spaCy model. Ensure 'symptom_ner_model' directory exists.") from e

# ------------------ Load Diseases ------------------
with open("diseases.json", "r", encoding="utf-8") as f:
    DISEASES = json.load(f).get("diseases", [])

# ------------------ Globals ------------------
COMMON_SYMPTOMS = list({s.lower() for d in DISEASES for s in d.get("symptoms", [])})
USER_CONTEXTS = {}
USER_STATE = {}
USER_HISTORY = {}
USER_STAGE = {}  # Tracks stage of the conversation

# ------------------ Utilities ------------------
def normalize(text: str) -> str:
    return text.lower().strip()

def extract_symptoms(text: str):
    doc = nlp(text)
    symptoms = [ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SYMPTOM"]
    return list(set(symptoms))

def diagnose(symptoms, top_k=3):
    results = []
    for disease in DISEASES:
        disease_symptoms = [s.lower() for s in disease.get("symptoms", [])]
        matched = []
        total_score = 0

        for user_symptom in symptoms:
            match = process.extractOne(user_symptom, disease_symptoms, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 60:  # Only consider relevant matches
                matched.append((user_symptom, match[0], match[1]))
                total_score += match[1]

        num_matches = len(matched)
        if num_matches == 0:
            continue  # Skip if no symptoms match

        # 💡 Composite scoring:
        score = (
            num_matches * 10 +             # Stronger score for more symptom matches
            (total_score / num_matches) / 10  # + average match quality
        )

        # 💥 Boost if >=3 matching symptoms (likely correct)
        if num_matches >= 3:
            score += 15

        results.append({
            **disease,
            "score": round(score, 2),
            "matched_symptoms": [m[1] for m in matched]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]



def get_disease_by_name(name: str):
    name = normalize(name)
    for d in DISEASES:
        if normalize(d["name"]) in name or name in normalize(d["name"]):
            return d
    return None

# ------------------ Routes ------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.json or {}
    user_input = (payload.get("input") or "").strip().lower()
    user_id = request.remote_addr or "local-user"

    if not user_input:
        return jsonify({"reply": "Please describe how you're feeling."})

    if user_id not in USER_CONTEXTS:
        USER_CONTEXTS[user_id] = [{"role": "assistant", "content": "Hi! How are you feeling today?"}]
    if user_id not in USER_HISTORY:
        USER_HISTORY[user_id] = []
    if user_id not in USER_STAGE:
        USER_STAGE[user_id] = "diagnosis"

    history = USER_CONTEXTS[user_id]
    history.append({"role": "user", "content": user_input})
    stage = USER_STAGE[user_id]
    last_disease = USER_STATE.get(user_id)

    if stage == "ask_remedies" and user_input in ["yes", "yeah", "yup", "ok", "okay"]:
        if last_disease:
            info = get_disease_by_name(last_disease)
            if info:
                USER_STAGE[user_id] = "ask_prevention"
                reply = (
                    f"🩺 Remedies for {info['name']}:\n"
                    + "\n".join(f"- {r}" for r in info.get("remedies", []))
                    + f"\n\nWould you like prevention tips for {info['name']}?"
                )
                history.append({"role": "assistant", "content": reply})
                return jsonify({"reply": reply})

    elif stage == "ask_prevention" and user_input in ["yes", "yeah", "yup", "ok", "okay"]:
        if last_disease:
            info = get_disease_by_name(last_disease)
            if info:
                USER_STAGE[user_id] = "done"
                reply = f"🛡️ Prevention for {info['name']}:\n{info.get('prevention', 'Not specified')}"
                history.append({"role": "assistant", "content": reply})
                return jsonify({"reply": reply})

    symptoms = extract_symptoms(user_input)

    if len(symptoms) >= 2:
        matches = diagnose(symptoms)
        if matches:
            top = matches[0]
            USER_STATE[user_id] = top["name"]
            USER_STAGE[user_id] = "ask_remedies"
            reply = (
                f"Based on your symptoms, you may have **{top['name']}**.\n"
                f"Would you like remedies for {top['name']}?"
            )
            USER_HISTORY[user_id].append({
                "ts": int(datetime.now().timestamp() * 1000),
                "input": user_input,
                "extracted": symptoms,
                "predictions": [{"name": d["name"], "confidence": d["score"]} for d in matches],
                "engine": "Local NER"
            })
            history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})

    for d in DISEASES:
        if normalize(d["name"]) in normalize(user_input):
            info = get_disease_by_name(d["name"])
            if info:
                USER_STATE[user_id] = info["name"]
                USER_STAGE[user_id] = "ask_remedies"
                reply = f"You mentioned **{info['name']}**. Would you like remedies for it?"
                history.append({"role": "assistant", "content": reply})
                return jsonify({"reply": reply})

    USER_STAGE[user_id] = "diagnosis"
    reply = "I couldn't detect enough symptoms. Could you describe how you're feeling in more detail?"
    history.append({"role": "assistant", "content": reply})
    return jsonify({"reply": reply})

@app.route("/api/history", methods=["GET"])
def get_history():
    user_id = request.remote_addr or "local-user"
    return jsonify({"history": USER_HISTORY.get(user_id, [])})

@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    user_id = request.remote_addr or "local-user"
    USER_HISTORY[user_id] = []
    return jsonify({"success": True})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
