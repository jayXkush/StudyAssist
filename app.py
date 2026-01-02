import streamlit as st
import traceback

from utils.ingestion import process_pdf
from chains.chat import get_qa_chain
from chains.flashcards import generate_summary, generate_flashcards

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 📂 Upload Study Material")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type="pdf"
    )

    if uploaded_file:
        if st.button("📥 Process Document", use_container_width=True):
            with st.spinner("Processing PDF..."):
                try:
                    vectorstore = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.success("✅ Document processed successfully!")
                except Exception as e:
                    st.error(str(e))

    st.divider()
    st.caption("📘 Documents are stored in memory for the current session")

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "💬 Tutor Chat",
    "📝 Topic Summary",
    "🃏 Flashcards"
])

# ==================================================
# 💬 CHAT TAB
# ==================================================
with tab1:
    st.markdown("## 📘 AI Tutor")

    # Containers to control layout
    chat_container = st.container()
    input_container = st.container()

    # Session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ---------- CHAT HISTORY ----------
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------- CHAT INPUT (FIXED AT BOTTOM) ----------
    with input_container:
        user_input = st.chat_input("Ask something about your PDF...")

    # ---------- HANDLE MESSAGE ----------
    if user_input:
        # User message
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        if "vectorstore" not in st.session_state:
                            st.error("Please upload and process a document first.")
                        else:
                            chain = get_qa_chain(st.session_state.vectorstore)
                            response = chain.invoke(user_input)

                            st.markdown(response)

                            st.session_state.messages.append(
                                {"role": "assistant", "content": response}
                            )

                    except Exception:
                        st.error("❌ Error occurred")
                        st.code(traceback.format_exc())

# ==================================================
# 📝 SUMMARY TAB
# ==================================================
with tab2:
    st.markdown("## 📝 Topic Summary")

    topic = st.text_input(
        "Enter a topic from the document",
        placeholder="e.g. Transformer Architecture"
    )

    if st.button("Generate Summary"):
        if topic.strip():
            if "vectorstore" not in st.session_state:
                st.error("Please upload and process a document first.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        st.markdown(generate_summary(topic, st.session_state.vectorstore))
                    except Exception:
                        st.error("Failed to generate summary")
        else:
            st.warning("Please enter a topic.")

# ==================================================
# 🃏 FLASHCARDS TAB
# ==================================================
with tab3:
    st.markdown("## 🃏 Flashcards")

    flash_topic = st.text_input(
        "Enter a topic",
        placeholder="e.g. Attention Mechanism",
        key="flash_topic"
    )

    if st.button("Generate Flashcards"):
        if flash_topic.strip():
            if "vectorstore" not in st.session_state:
                st.error("Please upload and process a document first.")
            else:
                with st.spinner("Generating flashcards..."):
                    try:
                        st.markdown(generate_flashcards(flash_topic, st.session_state.vectorstore))
                    except Exception:
                        st.error("Failed to generate flashcards")
        else:
            st.warning("Please enter a topic.")
