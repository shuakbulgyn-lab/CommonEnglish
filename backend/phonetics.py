"""Step 3: phonetic check against a British reference.

1. The reference text is turned into British IPA with espeak-ng (phonemizer, en-gb).
   (g2p_en gives American ARPAbet, so it is NOT used for British.)
2. The student's phonemes are recognised with a phoneme CTC model (wav2vec2 + espeak IPA).
3. Both sequences are aligned and every word is marked OK or with deviations.

Needs the espeak-ng program installed on the computer (see README).
"""
import os

PHONEME_MODEL = os.getenv("PHONEME_MODEL", "facebook/wav2vec2-lv-60-espeak-cv-ft")
STRESS = set("ˈˌ.")
MODIFIERS = set("ːˑ̃ʰʲʷ̩̯")
# sounds that count as the same for scoring (recogniser/espeak spelling differences)
EQUIV = {"ɡ": "g", "r": "ɹ", "ɚ": "ə", "ɐ": "ə", "ɫ": "l"}


def tokenize(ipa: str) -> list:
    """Split an IPA string into phoneme tokens (a length mark stays with its vowel)."""
    tokens = []
    for ch in ipa:
        if ch.isspace() or ch in STRESS:
            continue
        if ch in MODIFIERS and tokens:
            tokens[-1] += ch
        else:
            tokens.append(EQUIV.get(ch, ch))
    return tokens


def align_words(ref_words, hyp_tokens):
    """ref_words: list of (word, [ref tokens]); hyp_tokens: [tokens].
    Returns (per-word report, score 0-100)."""
    ref = []  # (token, word index)
    for wi, (_w, toks) in enumerate(ref_words):
        for t in toks:
            ref.append((t, wi))
    n, m = len(ref), len(hyp_tokens)
    INF = 10 ** 9
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if ref[i - 1][0] == hyp_tokens[j - 1] else 1
            D[i][j] = min(D[i - 1][j - 1] + sub, D[i - 1][j] + 1, D[i][j - 1] + 1)
    # traceback
    i, j = n, m
    status = [None] * n          # for each ref token: ("ok"|"sub"|"del", heard)
    extra = [[] for _ in ref_words]  # inserted sounds near a word
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (0 if ref[i - 1][0] == hyp_tokens[j - 1] else 1):
            ok = ref[i - 1][0] == hyp_tokens[j - 1]
            status[i - 1] = ("ok" if ok else "sub", hyp_tokens[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            status[i - 1] = ("del", "")
            i -= 1
        else:
            wi = ref[i - 1][1] if i > 0 else 0
            extra[wi].append(hyp_tokens[j - 1])
            j -= 1
    report, k = [], 0
    for wi, (w, toks) in enumerate(ref_words):
        devs, heard = [], []
        for t in toks:
            st, h = status[k]
            k += 1
            heard.append(h if st != "del" else "")
            if st != "ok":
                devs.append({"expected": t, "heard": h or "(missing)"})
        if extra[wi]:
            devs.append({"expected": "(nothing)", "heard": "".join(reversed(extra[wi]))})
        report.append({"word": w, "expected": "".join(toks), "heard": "".join(heard),
                       "ok": not devs, "deviations": devs})
    matches = sum(1 for s in status if s and s[0] == "ok")
    score = round(100 * matches / n, 1) if n else 0.0
    return report, score


def reference_words(text: str):
    from phonemizer import phonemize
    words = [w.strip(".,!?;:\"'()") for w in text.split()]
    words = [w for w in words if w]
    ipa = phonemize(words, language="en-gb", backend="espeak", strip=True, with_stress=False)
    return [(w, tokenize(p)) for w, p in zip(words, ipa)]


_proc = _model = None


def _load():
    global _proc, _model
    if _model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        _proc = Wav2Vec2Processor.from_pretrained(PHONEME_MODEL)
        _model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).eval()
    return _proc, _model


def recognise_phonemes(wav_path: str):
    import soundfile as sf
    import torch
    proc, model = _load()
    audio, sr = sf.read(wav_path)
    inputs = proc(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    ids = torch.argmax(logits, dim=-1)
    return tokenize(proc.batch_decode(ids)[0])


def analyze_pronunciation(wav_path: str, text: str) -> dict:
    ref = reference_words(text)
    hyp = recognise_phonemes(wav_path)
    words, score = align_words(ref, hyp)
    return {
        "british_reference_match_percent": score,
        "words": words,
        "note": ("Compared with a British (en-GB, espeak) reference. "
                 "The score shows how close the sounds are to that reference; it is an indicator, not a grade."),
    }
