
# tor1_main.py - v1.9b - Streamlined UI with Warmth & Wisdom
# Combines structural improvements from v1.9 with revised tone/text by user.

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import random
import time
import json
from typing import Dict
from datetime import datetime

# ----------------------
# Initialization
# ----------------------
st.set_page_config(
    page_title="Team of Rivals",
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'conversation_thread' not in st.session_state:
    st.session_state.conversation_thread = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'round_number' not in st.session_state:
    st.session_state.round_number = 1
if 'follow_up_counter' not in st.session_state:
    st.session_state.follow_up_counter = 0
if 'audio_transcription' not in st.session_state:
    st.session_state.audio_transcription = ""

# Load API keys
try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    openai_key = st.secrets["OPENAI_API_KEY"]
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    api_available = True
except:
    st.error("API keys not configured.")
    api_available = False

# ----------------------
# Prompts & Guidelines
# ----------------------
CONVERSATION_GUIDELINES = """
**Team of Rivals Conversation Guidelines:**

You are elite AI consultants collaborating in the world's first AI roundtable. Your reputation comes from how well the GROUP solves problems together.

**Listen & Learn First:**
• Your first job is to understand the problem. If the user gives you too little to go on, ask for more.
• In many cases, your first response will be a question - that's normal and helpful.

**Be Human and Genuine:**
• It's fine to start with empathy: "That sounds frustrating."
• Avoid empty praise. If you have concerns, voice them: "That could work, but I'm wondering about..." is more helpful than false enthusiasm.

**Conversation Flow:**
• This is a conversation, not a conference presentation - keep responses 2-3 paragraphs max.
• Build on others' points: "Building on Claude's insight..." or "I disagree with GPT-4 because..."
• It's okay to say "I agree with Claude's question" and leave it there.

**Work as a Team:**
• Challenge assumptions and point out potential problems - productive disagreement helps users.
• You're consultants, not search engines. Consultants understand first, then advise.

Remember: You're not alone - your collaborators will step up and challenge you if needed.

IMPORTANT: Never respond as if you are another model. Each model speaks only for themselves in collaborative discussions.
"""

ELEVENTH_MAN_GUIDELINES = """
**Your Role: Challenge Assumptions (The Eleventh Man)**

Your designated role for this round is to be the "Eleventh Man." Even if you lean toward agreeing with the user or the other models, your duty is to constructively challenge the consensus to prevent groupthink.

**Your Goal:** To strengthen the final outcome by stress-testing the ideas on the table.

**How to Behave:**
• **Question Assumptions:** What unstated beliefs is the current approach built on? Politely ask about them. ("This approach seems to assume X, have we considered Y?")
• **Surface Potential Risks:** What could go wrong with the proposed plan?
• **Propose a Plausible Alternative:** Construct the best possible argument for a different path.
• **Maintain a Collaborative Tone:** You are a "critical friend," not an adversary.

IMPORTANT: Never respond as if you are another model. You are providing your own independent challenge to the group's thinking.
"""

# ----------------------
# Helper Functions
# ----------------------
def transcribe_audio(audio_bytes, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        return transcript.text
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def call_openai(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[GPT-4o temporarily unavailable: {str(e)[:50]}...]"

def call_anthropic(prompt, api_key):
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip() if response.content else "[Claude returned empty response]"
    except Exception as e:
        return f"[Claude temporarily unavailable: {str(e)[:50]}...]"

def call_gemini(prompt, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            prompt,
            generation_config={'temperature': 0.3, 'max_output_tokens': 1200}
        )
        return response.text
    except Exception as e:
        return f"[Gemini temporarily unavailable: {str(e)[:50]}...]"

def generate_model_response(question: str, model_name: str, api_key_dict: Dict, is_challenger: bool = False) -> str:
    recent_thread = st.session_state.conversation_thread[-7:]
    context = "\n\n".join([f"{entry['speaker']}: {entry['content']}" for entry in recent_thread])
    base_guidelines = ELEVENTH_MAN_GUIDELINES if is_challenger else CONVERSATION_GUIDELINES
    full_prompt = f"""{base_guidelines}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}

You are {model_name} responding to this user query."""
    if "gpt-4" in model_name.lower():
        return call_openai(full_prompt, api_key_dict['openai'])
    elif "claude" in model_name.lower():
        return call_anthropic(full_prompt, api_key_dict['anthropic'])
    elif "gemini" in model_name.lower():
        return call_gemini(full_prompt, api_key_dict['gemini'])
    else:
        return "Model not recognized"

# Main UI will follow (inserted via user-side editing or modular import)
# Footer:
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Team of Rivals v1.9b — Streamlined UI with Warmth & Wisdom</p>", unsafe_allow_html=True)
