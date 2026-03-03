import os
import requests
import streamlit as st

# Set page config for premium look
st.set_page_config(
    page_title="Help Desk Voice Bot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for glassmorphism and modern UI per requirements
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Headers and Text */
    h1, h2, h3 {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Center text alignment for title */
    .title-container {
        text-align: center;
        padding-bottom: 2rem;
    }

    /* Primary color override for buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: none;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* File uploader styling */
    .css-1n76uvr {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

API_BASE = os.getenv("API_URL", "http://localhost:8000/api").rstrip("/api") + "/api"

st.markdown(
    "<div class='title-container'><h1>🎙️ Premium IT Help Desk Bot</h1>"
    "<p>Powered by RAG, Gemma 3, and Local Voice Models</p></div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🗣️ Voice", "💬 Text Chat", "📄 Ingest Documents"])

with tab1:
    st.write("### Record or upload your query")
    st.info(
        "Upload a WAV or MP3 file. The backend will transcribe it, run RAG, and return a spoken response."
    )
    audio_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"], key="voice_upload")
    if audio_file is not None:
        st.audio(audio_file, format="audio/wav" if audio_file.name and audio_file.name.lower().endswith(".wav") else "audio/mpeg")
        if st.button("Process voice request", key="voice_btn"):
            with st.spinner("Transcribing, querying RAG, and synthesizing response..."):
                try:
                    audio_bytes = audio_file.read()
                    # Preserve filename so backend can detect format
                    fname = audio_file.name or "audio.wav"
                    files = {"audio_file": (fname, audio_bytes, audio_file.type or "audio/wav")}
                    response = requests.post(f"{API_BASE}/voice", files=files, timeout=60)
                    if response.status_code == 200:
                        st.success("Response received!")
                        user_text = response.headers.get("X-User-Text", "—")
                        bot_text = response.headers.get("X-Bot-Text", "—")
                        bot_audio_bytes = response.content
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("🗣️ You said")
                            st.write(user_text)
                        with col2:
                            st.subheader("🤖 Bot says")
                            st.write(bot_text)
                        st.subheader("🎧 Listen to response")
                        st.audio(bot_audio_bytes, format="audio/mp3")
                    else:
                        st.error(f"Backend error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to process: {e}")

with tab2:
    st.write("### Text chat with the help desk")
    st.info("Type your question; the bot will answer using the RAG knowledge base.")
    msg = st.text_input("Your question", key="chat_input", placeholder="e.g. How do I reset my password?")
    if st.button("Send", key="chat_btn") and msg:
        with st.spinner("Querying RAG..."):
            try:
                response = requests.post(
                    f"{API_BASE}/chat",
                    json={"message": msg},
                    timeout=30,
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success("Reply:")
                    st.write(data.get("response", ""))
                else:
                    st.error(f"Backend error: {response.text}")
            except Exception as e:
                st.error(f"Failed: {e}")

with tab3:
    st.write("### Ingest documents into the knowledge base")
    st.info("Upload PDF or TXT files. They will be split and added to the vector store for RAG.")
    ingest_file = st.file_uploader(
        "Upload document (PDF or TXT)",
        type=["pdf", "txt"],
        key="ingest_upload",
    )
    if ingest_file is not None and st.button("Ingest document", key="ingest_btn"):
        with st.spinner("Splitting and adding to vector store..."):
            try:
                files = {"file": (ingest_file.name, ingest_file.read(), ingest_file.type or "application/octet-stream")}
                response = requests.post(f"{API_BASE}/ingest", files=files, timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Ingested {data.get('chunks_added', 0)} chunks from {data.get('filename', '')}.")
                else:
                    err = response.json() if response.headers.get("content-type", "").startswith("application/json") else {"error": response.text}
                    st.error(err.get("error", response.text))
            except Exception as e:
                st.error(f"Failed: {e}")
