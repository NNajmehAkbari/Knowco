# app.py
import os
import streamlit as st
import numpy as np

# --- Your libs ---
import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.preview.generative_models import GenerativeModel
from utils.helper_functions import (
    process_pdf_folder,
    clean_excerpt,
    chunk_document,
    embed_chunks,
    retrieve_top_k,
    embedding_functions,
    prompts_call,
)

# (Optional) ElevenLabs TTS
try:
    from elevenlabs.client import ElevenLabs
    ELEVEN_AVAILABLE = True
except Exception:
    ELEVEN_AVAILABLE = False

# ----------------------------
# UI CONFIG
# ----------------------------
st.set_page_config(page_title="KnowCoach", page_icon="🎓", layout="centered")

st.title("🎓 KnowCoach – Document-grounded Coach")
st.caption("Coach asks questions generated from your company documents (RAG).")

# ----------------------------
# SIDEBAR: CONFIG
# ----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # GCP / Vertex
    json_path   = st.text_input("GOOGLE_APPLICATION_CREDENTIALS (JSON path)", value=os.getenv("GOOGLE_APPLICATION_CREDENTIALS",""))
    project_id  = st.text_input("GCP Project ID", value=os.getenv("GCP_PROJECT",""))
    location    = st.text_input("Vertex Location (e.g., us-central1, europe-west1)", value=os.getenv("VERTEX_LOCATION","us-central1"))

    pdf_folder  = st.text_input("PDF Folder Path", value=os.getenv("PDF_FOLDER","/path/to/pdfs"))
    image_dir   = st.text_input("Image Output Folder", value=os.getenv("IMAGE_DIR","./_coach_images"))

    # Models
    gen_model_name = st.text_input("Generative Model", value="gemini-2.0-flash-001")
    embed_model_name = st.text_input("Embedding Model", value="text-embedding-005")

    # (Optional) TTS
    tts_enabled = st.checkbox("🔊 Speak coach questions (ElevenLabs)", value=False, disabled=not ELEVEN_AVAILABLE)
    coach_voice = st.text_input("ElevenLabs Coach Voice ID", value="EXAVITQu4vr4xnSDxMaL", disabled=not tts_enabled)
    eleven_api  = st.text_input("ElevenLabs API Key", type="password", value=os.getenv("ELEVEN_API_KEY",""), disabled=not tts_enabled)

    st.markdown("---")
    go_btn = st.button("🚀 Initialize & Process Documents", type="primary")

# ----------------------------
# HELPERS
# ----------------------------
def init_vertex(json_path: str, project_id: str, location: str):
    if json_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_path
    vertexai.init(project=project_id, location=location)

def build_corpus(pdf_folder: str, image_dir: str, image_prompt_name: str, gen_model):
    # Extract text, tables, images → combine as one corpus
    process_pdf = process_pdf_folder(pdf_folder, image_dir, image_prompt_name, gen_model)
    full_text_parts = []
    for pdf_data in process_pdf.values():
        for page in pdf_data:
            full_text_parts.append(page.get("text", ""))
            full_text_parts.extend(page.get("tables", []))
            full_text_parts.extend(page.get("images", []))
    return clean_excerpt("\n".join(full_text_parts))

def generate_coach_question(multimodal_model, embedding_model, chunk_cluster, vectors):
    """
    RAG-driven question generation:
    1) retrieve small set of relevant chunks using a generic training query
    2) ask model to craft a short question from that context
    """
    top_chunks = retrieve_top_k(
        "Generate a practical training question about the company content.",
        embedding_model,
        chunk_cluster,
        vectors,
        k=3
    )
    context = "\n".join([c for c, _score in top_chunks])

    prompt = f"""
    You are a professional coach.
    Based on the following company content, generate ONE short, clear question
    that a trainee should be able to answer. Only output the question.

    Context:
    {context}
    """
    rsp = multimodal_model.generate_content(prompt)
    return rsp.text.strip(), context

def answer_with_rag(multimodal_model, embedding_model, chunk_cluster, vectors, question: str):
    top_chunks = retrieve_top_k(question, embedding_model, chunk_cluster, vectors, k=3)
    ctx = "\n".join([c for c, _ in top_chunks])
    prompt = f"""Answer the question strictly using the provided context.

Context:
{ctx}

Question: {question}
"""
    rsp = multimodal_model.generate_content(prompt)
    return rsp.text.strip(), ctx

def speak_eleven(text: str, api_key: str, voice_id: str):
    if not ELEVEN_AVAILABLE:
        return "ElevenLabs SDK not installed."
    try:
        client = ElevenLabs(api_key=api_key)
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128"
        )
        # Streamlit audio expects bytes-like object
        audio_bytes = b"".join(chunk for chunk in audio if chunk)
        st.audio(audio_bytes, format="audio/mp3")
        return None
    except Exception as e:
        return str(e)

# ----------------------------
# STATE
# ----------------------------
if "ready" not in st.session_state:
    st.session_state.ready = False
    st.session_state.texts = None
    st.session_state.vectors = None
    st.session_state.chunks = None
    st.session_state.multimodal = None
    st.session_state.embed_model = None

# ----------------------------
# INIT & BUILD
# ----------------------------
if go_btn:
    try:
        with st.spinner("Initializing Vertex AI…"):
            init_vertex(json_path, project_id, location)
            multimodal = GenerativeModel(gen_model_name)
            embed_model = TextEmbeddingModel.from_pretrained(embed_model_name)

        with st.spinner("Processing PDFs, extracting text/images…"):
            image_prompt = prompts_call("image_description_prompt")
            corpus = build_corpus(pdf_folder, image_dir, image_prompt, multimodal)

        with st.spinner("Chunking document…"):
            chunks = chunk_document(
                corpus,
                method="cluster",
                embedding_function=embedding_functions,
            )

        with st.spinner("Embedding chunks…"):
            texts, vectors, failures = embed_chunks(chunks, embed_model)
            vectors_np = np.stack(vectors) if len(vectors) else np.zeros((0, 768), dtype=np.float32)

        st.session_state.ready = True
        st.session_state.texts = texts
        st.session_state.vectors = vectors_np
        st.session_state.chunks = texts
        st.session_state.multimodal = multimodal
        st.session_state.embed_model = embed_model

        st.success(f"Ready! Chunks: {len(texts)}, Failed embeddings: {len(failures)}")

    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.stop()

# ----------------------------
# MAIN UI
# ----------------------------
if st.session_state.ready:
    tab1, tab2 = st.tabs(["🧠 Coach", "📄 Document Q&A"])

    # ---------- Coach Tab ----------
    with tab1:
        st.subheader("Coach Mode")
        st.write("Click **Ask** to have the coach generate a question from your documents.")
        colA, colB = st.columns(2)
        with colA:
            ask_btn = st.button("🗣️ Ask (RAG)")

        if ask_btn:
            q, ctx = generate_coach_question(
                st.session_state.multimodal,
                st.session_state.embed_model,
                st.session_state.chunks,
                st.session_state.vectors,
            )
            st.session_state["last_question"] = q
            st.session_state["last_context"] = ctx

        if "last_question" in st.session_state:
            st.markdown("**Coach question:**")
            st.info(st.session_state["last_question"])

            if tts_enabled and eleven_api:
                err = speak_eleven(st.session_state["last_question"], eleven_api, coach_voice)
                if err:
                    st.warning(f"TTS error: {err}")

            st.markdown("**Your answer (freeform):**")
            user_answer = st.text_area("Type your answer here…", key="coach_answer", height=120, label_visibility="collapsed")
            col1, col2 = st.columns(2)
            with col1:
                check_btn = st.button("✅ Check with Document (RAG)")
            with col2:
                new_q_btn = st.button("🔁 New Question")

            if check_btn:
                ans, used_ctx = answer_with_rag(
                    st.session_state.multimodal,
                    st.session_state.embed_model,
                    st.session_state.chunks,
                    st.session_state.vectors,
                    st.session_state["last_question"]
                )
                with st.expander("Context used"):
                    st.code(used_ctx[:2000])
                st.success("**Coach reference answer:**")
                st.write(ans)

            if new_q_btn:
                st.session_state.pop("last_question", None)
                st.session_state.pop("last_context", None)
                st.experimental_rerun()

    # ---------- Document Q&A Tab ----------
    with tab2:
        st.subheader("Ask the document directly")
        user_q = st.text_input("Your question")
        go_ans = st.button("🔎 Retrieve & Answer")
        if go_ans and user_q.strip():
            ans, used_ctx = answer_with_rag(
                st.session_state.multimodal,
                st.session_state.embed_model,
                st.session_state.chunks,
                st.session_state.vectors,
                user_q
            )
            with st.expander("Context used"):
                st.code(used_ctx[:2000])
            st.markdown("**Answer:**")
            st.write(ans)

