# tor1_main.py

import streamlit as st
from app.components.ui import show_header, show_round_controls, show_transcript
from app.logic.round_manager import handle_round_logic
from app.utils.session_state import initialize_session_state

# ----------------------
# Initialization
# ----------------------
st.set_page_config(page_title="Team of Rivals", layout="wide")
initialize_session_state()

# ----------------------
# Header + Tagline
# ----------------------
st.markdown("<h1 style='margin-bottom: 0;'>Team of Rivals</h1>", unsafe_allow_html=True)
st.markdown("*ChatGPT, Claude, Gemini. Three top AI minds. One collaborative challenge.*")

# ----------------------
# Collapsible Disclaimer Panel (Default: Open)
# ----------------------
with st.expander("🔍 Why Not Just Use the Free Model Sites?", expanded=True):
    st.markdown("👋 **All-Too-Candid Opening Disclaimer**")
    st.markdown("""
    When Mel Brooks was asked at 90 whether he wore boxers or briefs, he answered: **Depends.**

    If your question is simple, the answer is no. You’ll get faster, stronger replies using the free tools at 
    [chatgpt.com](https://chatgpt.com), [claude.ai](https://claude.ai), or [gemini.google.com](https://gemini.google.com). 
    This app connects to those same models via API, and — for complicated reasons — their answers are often weaker.

    But here's a tip: **Ask your question in one model, then paste the answer into another and ask, “Does this seem right?”** 
    You’ll catch hallucinations and sharpen your thinking.
    """)

    st.markdown("### 🤔 So What’s This Tool For?")
    st.markdown("""
    **Complicated, layered questions.** The kind that benefit from disagreement, reflection, and synthesis. 
    Here, the models aren’t just answering — they’re working together. Sometimes it adds up to more than the sum of the parts.

    So is this app truly better for complex, layered questions than just using one model on its own?  
    **We think so — but help us find out.**
    """)

# ----------------------
# Challenge Input
# ----------------------
st.markdown("🗣️ **Share Your Challenge — talk it out or type it up**")

user_prompt = st.text_area(
    "Think out loud, paste in a problem, or describe a question you'd like the team to explore...",
    key="user_prompt_input",
    height=150,
)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Ask the Team", type="primary", use_container_width=True, key="submit_btn"):
        st.session_state.start_round = True
        st.session_state.transcript.append({"role": "user", "content": user_prompt})

# ----------------------
# Controls & Transcript
# ----------------------
show_round_controls()
handle_round_logic()
show_transcript()

# ----------------------
# Sidebar (Minimal + Dev Toggle)
# ----------------------
with st.sidebar:
    st.markdown("### About this Tool")
    st.write("Team of Rivals brings multiple AI models together to tackle your toughest questions.")
    st.write("Developed as an experimental collaboration platform for comparative AI reasoning.")

    # Dev-only Quick Mode toggle (currently does nothing)
    if st.checkbox("⚙️ Enable Quick Mode (dev only)", value=False):
        st.info("Quick Mode toggle is here for future development — currently inactive.")

# ----------------------
# Footer / Version Label
# ----------------------
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Team of Rivals v1.1 — Quick Mode Removed, Collaborative Only</p>", unsafe_allow_html=True)
