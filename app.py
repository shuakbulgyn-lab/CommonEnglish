"""Step 4: Streamlit interface.  Run:  streamlit run app.py"""
import requests
import streamlit as st

st.set_page_config(page_title="Accent Checker", page_icon="🎙️", layout="centered")
st.title("🎙️ Accent Checker")
st.caption("Record a sentence and see how close it is to British English. This is a learning indicator, not a grade.")

backend = st.sidebar.text_input("Backend address", "http://127.0.0.1:8000")
st.sidebar.info("Use only with the permission of the speaker. Recordings are deleted right after the check.")

text = st.text_input("Sentence to read", "The path to the park is far")
if hasattr(st, "audio_input"):
    audio = st.audio_input("Press the microphone and read the sentence")
else:
    audio = st.file_uploader("Upload a recording (WAV, MP3, M4A)", type=["wav", "mp3", "m4a", "ogg"])

if audio is not None and st.button("Check my accent", type="primary"):
    with st.spinner("Analysing the recording..."):
        try:
            r = requests.post(
                f"{backend}/check-accent",
                files={"audio": (getattr(audio, "name", "recording.wav"), audio.getvalue())},
                data={"reference_text": text},
                timeout=300,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            st.error(f"Could not get a result: {e}")
            st.stop()

    if data.get("stub"):
        st.info(data["message"])
        st.stop()

    acc = data["accents"]
    st.subheader("Accent probabilities")
    c1, c2 = st.columns(2)
    c1.metric("British accent probability", f'{acc["british_percent"]}%')
    if "pronunciation" in data:
        c2.metric("British RP match (sounds)", f'{data["pronunciation"]["british_reference_match_percent"]}%')
    for name, p in acc["four_accents"].items():
        st.write(name)
        st.progress(min(int(p), 100), text=f"{p}%")
    st.caption(acc["note"])

    if "pronunciation" in data:
        st.subheader("Words to practise")
        rows = []
        for w in data["pronunciation"]["words"]:
            rows.append({
                "Word": w["word"],
                "Expected (British IPA)": w["expected"],
                "You said": w["heard"] or "-",
                "Status": "✅ OK" if w["ok"] else "⚠️ " + ", ".join(f'{d["expected"]}→{d["heard"]}' for d in w["deviations"]),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(data["pronunciation"]["note"])
