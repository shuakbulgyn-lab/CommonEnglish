"""Step 2: accent classification with a Hugging Face / SpeechBrain model.

Default model: CommonAccent (wav2vec2 XLS-R) that classifies 16 English accents.
Check the model name on huggingface.co before use and change MODEL_ID if needed.
The classifier gives a probability for each accent, so we can report all four
accents of the project (British, American, Australian, Canadian), not only British.
"""
import os

import torch

MODEL_ID = os.getenv("ACCENT_MODEL", "Jzuluaga/accent-id-commonaccent_xlsr-en-english")
# model label -> name shown to the student
TARGETS = {"england": "British", "us": "American", "australia": "Australian", "canada": "Canadian"}

_clf = None


def _load():
    global _clf
    if _clf is None:
        from speechbrain.pretrained.interfaces import foreign_class
        _clf = foreign_class(
            source=MODEL_ID,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir="pretrained_models/accent_id",
        )
    return _clf


def classify_accent(wav_path: str) -> dict:
    clf = _load()
    out_prob, _score, _idx, _label = clf.classify_file(wav_path)
    probs = torch.softmax(out_prob.squeeze(), dim=-1).tolist()
    ind2lab = clf.hparams.label_encoder.ind2lab
    all_probs = {ind2lab[i]: p for i, p in enumerate(probs)}
    result = {
        "british_percent": round(100 * all_probs.get("england", 0.0), 1),
        "four_accents": {name: round(100 * all_probs.get(key, 0.0), 1) for key, name in TARGETS.items()},
        "top_label": max(all_probs, key=all_probs.get),
        "note": "Probabilities from a classifier over 16 English accents. This is an indicator, not a grade.",
    }
    return result
