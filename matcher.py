import spacy

nlp = spacy.load("en_core_web_sm")

COMMON_SYMPTOMS = [
    "fever", "cough", "fatigue", "headache", "nausea", "vomiting",
    "chest pain", "shortness of breath", "joint pain", "rash",
    "dizziness", "sore throat", "diarrhea", "abdominal pain",
    "loss of appetite", "sweating", "weight loss", "chills",
    "sensitivity to light", "stiff neck", "back pain", "itchy throat",
    "itchy eyes", "sneezing", "watery eyes", "runny nose", "nasal congestion"
]

def extract_symptoms(text: str):
    """Extract symptoms from free text using spaCy + keyword matching"""
    doc = nlp(text)
    extracted = []

    for token in doc:
        if token.text.lower() in COMMON_SYMPTOMS:
            extracted.append(token.text.lower())

    for chunk in doc.noun_chunks:
        if chunk.text.lower() in COMMON_SYMPTOMS:
            extracted.append(chunk.text.lower())

    return list(set(extracted))
