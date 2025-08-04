# tor1_main.py - v1.7e — Undoing snafu — you know who you are!

import streamlit as st
import google.generativeai as genai
import os
from typing import Dict

# ----------------------
# API Key Setup
# ----------------------

def get_api_keys() -> Dict[str, str]:
    return {
        "gemini": os.getenv("GEMINI_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", "")
    }

# ----------------------
# Model Response Function
# ----------------------

def generate_model_response(question: str, model_name: str, api_key_dict: Dict, round_num: int = 1) -> str:
    placeholder = "🔧 [Model response placeholder]"
    if model_name == "Claude":
        # Claude logic placeholder
        return placeholder
    elif model_name == "ChatGPT":
        # ChatGPT logic placeholder
        return placeholder
    elif model_name == "Gemini":
        # Gemini logic placeholder
        return placeholder
    else:
        return "Unknown model"

# ----------------------
# Page Layout & UI
# ----------------------

st.set_page_config(page_title="Team of Rivals", layout="wide")

# ---- Header ----
st.markdown("<h1 style='margin-bottom: 0;'>Team of Rivals</h1>", unsafe_allow_html=True)
st.markdown("*ChatGPT, Claude, Gemini. Three top AI minds. One collaborative challenge.*")

# ---- New Intro (Shortened, Combined) ----
with st.expander("🧠 Introduction to Team of Rivals", expanded=True):
    st.markdown("""
**Your personal strategy team — now with fewer hallucinations.**

For straightforward tasks or quick questions, a single model may be exactly what you need.  
But when the problem is messy or layered, bringing multiple models together can help challenge hidden assumptions, flag blind spots, and reduce the risk of getting made-up answers.

The real benefit is how ToR can push your thinking deeper.  
This isn’t about passively accepting AI answers — it’s about doing the work as the leader of a strong and capable team, and without a single yes-man in the room.  
And along the way, you’ll likely pick up a better sense of how large language models think, what they miss, and how they complement each other.
""")

# ---- Use Case Columns ----
st.markdown("### 🤹 When To Use ToR — and When Not To")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### ✅ When ToR Helps Most")
    st.markdown("""
- Complicated, layered questions  
- Tradeoffs, strategy, dilemmas  
- Need for reflection or synthesis  
- Wanting to see different viewpoints  
- Curious how models *disagree*
""")

with col2:
    st.markdown("#### 🚫 When a Single Model Is Faster")
    st.markdown("""
- Simple facts or definitions  
- Quick rewrites or summaries  
- Repetitive editing tasks  
- Straightforward Q&A  
- You're in a hurry
""")

st.markdown("---")

# ---- User Prompt ----
st.subheader("📢 Share Your Challenge — talk it out or type it in")

question = st.text_area("What would you like the team to tackle?", height=150)

# ---- Model Selection ----
st.markdown("### 🧠 Choose Your Team")
models_selected = st.multiselect("Pick which models to include:", ["ChatGPT", "Claude", "Gemini"], default=["ChatGPT", "Claude", "Gemini"])

# ---- API Keys ----
api_keys = get_api_keys()

# ---- Response Trigger ----
if st.button("🧠 Run Team of Rivals"):
    if not question.strip():
        st.warning("Please enter a challenge or question.")
    elif not models_selected:
        st.warning("Please select at least one model.")
    else:
        st.markdown("---")
        st.subheader("🤖 Team Responses")
        for model in models_selected:
            with st.expander(f"{model}'s Response", expanded=True):
                response = generate_model_response(question, model, api_keys)
                st.markdown(response)

# ---- Footer ----
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Team of Rivals v1.7e — Undoing snafu — you know who you are!</p>", unsafe_allow_html=True)
