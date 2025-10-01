import spacy
from spacy.training.example import Example
from pathlib import Path

# Load a blank English NLP model
nlp = spacy.blank("en")

# Add the NER pipe
if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")

# ✅ Sample TRAIN_DATA with multiple symptoms
TRAIN_DATA = [
    ("I have fever and cough", {"entities": [(7, 12, "SYMPTOM"), (17, 22, "SYMPTOM")]}),
    ("Experiencing headache and nausea", {"entities": [(13, 21, "SYMPTOM"), (26, 32, "SYMPTOM")]}),
    ("Shortness of breath and fatigue bothering me", {"entities": [(0, 19, "SYMPTOM"), (24, 31, "SYMPTOM")]}),
    ("I feel weak and dizzy", {"entities": [(7, 11, "SYMPTOM"), (16, 21, "SYMPTOM")]}),
    ("I’ve been feeling tired and have loss of smell", {"entities": [(18, 23, "SYMPTOM"), (33, 47, "SYMPTOM")]}),
]

# Add labels
for _, annotations in TRAIN_DATA:
    for ent in annotations["entities"]:
        ner.add_label(ent[2])

# Disable other pipeline components
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for epoch in range(20):
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], losses=losses)
        print(f"Epoch {epoch+1} Losses: {losses}")

# ✅ Save model
output_dir = Path("symptom_ner_model")
nlp.to_disk(output_dir)
print(f"✅ Model saved to: {output_dir.resolve()}")

# ✅ Test on unseen input
test_text = "I feel very dizzy and have blurred vision"
doc = nlp(test_text)
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
