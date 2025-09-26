import os
import json
import spacy
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from flask_session import Session
from flask_cors import CORS
from fuzzywuzzy import process, fuzz
from openai import OpenAI

# ------------------ Env & Clients ------------------
load_dotenv()
client = OpenAI()

# ------------------ Flask Setup ------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "super-secret-key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
CORS(app)

# ------------------ Load spaCy Trained Model ------------------
nlp = spacy.load("symptom_ner_model")

# ------------------ Load Diseases ------------------
with open("diseases.json", "r", encoding="utf-8") as f:
    DISEASES = json.load(f)["diseases"]

# ------------------ Globals ------------------
COMMON_SYMPTOMS = list({s.lower() for d in DISEASES for s in d.get("symptoms", [])})
USER_CONTEXTS = {}
USER_STATE = {}  # Tracks last disease per user

# ------------------ Utilities ------------------
def normalize(text: str) -> str:
    return text.lower().strip()

def extract_symptoms(text: str):
    doc = nlp(text)
    symptoms = [ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SYMPTOM"]
    return list(set(symptoms))

def diagnose(symptoms, top_k=3):
    results = []
    for d in DISEASES:
        disease_symptoms = [s.lower() for s in d.get("symptoms", [])]
        scores = []
        for s in symptoms:
            match = process.extractOne(s, disease_symptoms, scorer=fuzz.token_sort_ratio)
            if match:
                scores.append(match[1])
        good = sum(1 for sc in scores if sc >= 70)
        avg = sum(scores) / len(scores) if scores else 0
        composite = good * 10 + avg / 10
        results.append({**d, "score": round(composite, 2)})
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
    user_input = (payload.get("input") or "").strip()
    user_id = request.remote_addr or "local-user"

    # Initialize session
    if user_id not in USER_CONTEXTS:
        USER_CONTEXTS[user_id] = [
            {
                "role": "system",
                "content": (
                    "You are Healtho, a friendly healthcare assistant. "
                    "Greet the user, ask how they’re feeling, extract symptoms, "
                    "offer possible conditions from the user's symptom list, and "
                    "share remedies/prevention from the local database when asked. "
                    "Be empathetic, concise, and natural."
                ),
            },
            {"role": "assistant", "content": "Hi! How are you feeling today?"}
        ]

    history = USER_CONTEXTS[user_id]
    history.append({"role": "user", "content": user_input})

    # ✅ Handle "yes" if bot just asked about remedies
    if user_input.lower() in ["yes", "yeah", "yup", "ok", "okay"]:
        last_disease = USER_STATE.get(user_id)
        if last_disease:
            info = get_disease_by_name(last_disease)
            if info:
                reply = (
                    f"🩺 Remedies for {info['name']}:\n"
                    + "\n".join(f"- {r}" for r in info.get("remedies", []))
                    + f"\n\nPrevention: {info.get('prevention', 'Not specified')}"
                )
                history.append({"role": "assistant", "content": reply})
                return jsonify({"reply": reply})

    # ✅ Extract symptoms
    symptoms = extract_symptoms(user_input)

    # ✅ If 2+ symptoms, diagnose
    if len(symptoms) >= 2:
        matches = diagnose(symptoms)
        if matches:
            names = ", ".join(d["name"] for d in matches)
            top = matches[0]
            USER_STATE[user_id] = top["name"]  # store for "yes" follow-up
            reply = (
                f"Based on what you've shared, possible conditions are: {names}.\n\n"
                f"Most likely: **{top['name']}**\n"
                f"Treatment: {top.get('treatment', 'N/A')}\n"
                f"Would you like remedies or prevention tips for {top['name']}?"
            )
            history.append({"role": "assistant", "content": reply})
            return jsonify({"reply": reply})

    # ✅ If direct disease mention (e.g., "Chickenpox")
    for d in DISEASES:
        if normalize(d["name"]) in normalize(user_input):
            info = get_disease_by_name(d["name"])
            if info:
                reply = (
                    f"🩺 Remedies for {info['name']}:\n"
                    + "\n".join(f"- {r}" for r in info.get("remedies", []))
                    + f"\n\nPrevention: {info.get('prevention', 'Not specified')}"
                )
                history.append({"role": "assistant", "content": reply})
                return jsonify({"reply": reply})

    # ❌ Not enough symptoms
    if len(symptoms) < 2:
        reply = "I couldn't detect enough symptoms to offer a diagnosis. Could you please describe how you're feeling in more detail?"
        history.append({"role": "assistant", "content": reply})
        return jsonify({"reply": reply})

    # 🤖 Fallback to GPT-3.5
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0.8,
            messages=history
        )
        bot_reply = response.choices[0].message.content
    except Exception as e:
        print("❌ GPT API Error:", str(e))
        bot_reply = (
            "⚠️ Sorry, I'm unable to connect to the AI service right now. "
            "You can still tell me your symptoms, and I’ll try my best to assist you!"
        )

    history.append({"role": "assistant", "content": bot_reply})
    return jsonify({"reply": bot_reply})

# ------------------ Run App ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
