# tor1_main.py - v1.8 - Clean UI + Code Quality Fixes
# This version removes the sidebar, implements streamlined UI, and fixes token limit issues

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import random
import time
from typing import Dict
from datetime import datetime

# ----------------------
# Configuration
# ----------------------
CONTEXT_HISTORY_TURNS = 5  # Reduced from 7 to prevent token limit issues
MODELS_CONFIG = {
    "GPT-4": {"call_func": "call_openai", "icon": "🤖"},
    "Claude": {"call_func": "call_anthropic", "icon": "🧠"},
    "Gemini": {"call_func": "call_gemini", "icon": "💎"}
}

# ----------------------
# Page Layout & UI
# ----------------------
st.set_page_config(page_title="Team of Rivals", layout="wide")

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
except Exception as e:
    st.error(f"API keys not configured: {e}")
    api_available = False

# ---- Header ----
st.markdown("<h1 style='margin-bottom: 0;'>Team of Rivals</h1>", unsafe_allow_html=True)
st.markdown("*ChatGPT, Claude, Gemini. Three top AI minds. One collaborative challenge.*")

# ---- New Intro (Shortened, Combined) ----
with st.expander("🧠 Introduction to Team of Rivals", expanded=True):
    st.markdown("""
**Your personal strategy team — now with fewer hallucinations.**  
For straightforward tasks or quick questions, a single model may be exactly what you need. When the problem is messy or layered, bringing multiple models together can help challenge hidden assumptions, flag blind spots, and reduce the risk of getting made-up answers. But the real benefit might be how ToR can push your thinking deeper.
""")

# ---- Use Case Columns ----
st.markdown("### When To Use ToR")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### When ToR Helps Most")
    st.markdown("""
- Complicated, layered questions  
- Need for reflection or synthesis  
- Wanting to see different viewpoints
""")

with col2:
    st.markdown("#### When Not To")
    st.markdown("""
- Simple facts or definitions  
- Quick rewrites or summaries  
- Repetitive editing tasks
""")

st.markdown("---")

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
    """Transcribe audio using OpenAI Whisper"""
    try:
        if not audio_bytes:
            return "Error: No audio data received"
            
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
    """Generate a model's response with conversation context and guidelines"""
    recent_thread = st.session_state.conversation_thread[-CONTEXT_HISTORY_TURNS:]
    context = "\n\n".join([f"{entry['speaker']}: {entry['content']}" for entry in recent_thread])
    
    base_guidelines = ELEVENTH_MAN_GUIDELINES if is_challenger else CONVERSATION_GUIDELINES
    
    full_prompt = f"""{base_guidelines}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}

You are {model_name} responding to this user query."""

    # Use centralized model configuration
    if model_name in MODELS_CONFIG:
        call_func_name = MODELS_CONFIG[model_name]["call_func"]
        if call_func_name == "call_openai":
            return call_openai(full_prompt, api_key_dict['openai'])
        elif call_func_name == "call_anthropic":
            return call_anthropic(full_prompt, api_key_dict['anthropic'])
        elif call_func_name == "call_gemini":
            return call_gemini(full_prompt, api_key_dict['gemini'])
    
    return "Model not recognized"

# ----------------------
# Main UI
# ----------------------
st.subheader("📢 Share Your Challenge -- talk it out or type it in")
st.markdown("**Think out loud:** Complex problems often are best explained by rambling. Don't worry about being precise - the consultants will ask follow-up questions!")

# Audio input
audio_input = st.audio_input("🎤 Record your challenge (captures nuance and context)")
if audio_input is not None and api_available:
    with st.spinner("Transcribing your audio..."):
        audio_bytes = audio_input.read()
        if not audio_bytes:
            st.warning("Could not read audio input. Please try recording again.")
        else:
            transcription = transcribe_audio(audio_bytes, openai_key)
            st.session_state.audio_transcription = transcription
            
            if transcription and not transcription.startswith("Error"):
                st.success("🎤 Audio transcribed!")
                with st.expander("📝 What I heard", expanded=True):
                    st.write(transcription)

# Text input
user_problem = st.text_area(
    "Or type your challenge here:",
    placeholder="Examples:\n- I have some Python code that feels messy\n- Need help planning a difficult team conversation\n- Trying to decide between two strategic directions",
    height=120,
    key="problem_input"
)

# Combine inputs
combined_input = ""
if user_problem:
    combined_input += user_problem
if st.session_state.audio_transcription and not st.session_state.audio_transcription.startswith("Error"):
    if combined_input:
        combined_input += "\n\n[From voice input:] " + st.session_state.audio_transcription
    else:
        combined_input = st.session_state.audio_transcription

if st.button("🚀 Ask the Team", type="primary", disabled=not combined_input.strip() or not api_available):
    st.session_state.conversation_thread = [{
        "speaker": "User",
        "content": combined_input,
        "timestamp": "Initial request",
        "time": datetime.now()
    }]
    st.session_state.session_active = True
    st.session_state.round_number = 1
    st.rerun()

# Challenge Mode Callout
st.info("🕵️ **Challenge Assumptions Mode:** Activate this mode once a conversation gets going to assign one AI as a bit of a devil's advocate - they'll constructively question the group's consensus to prevent groupthink.")

# ----------------------
# Active Consultation Display
# ----------------------
if st.session_state.session_active:
    st.markdown("---")
    st.subheader("💬 Consultation in Progress")

    for entry in st.session_state.conversation_thread:
        with st.container():
            if entry['speaker'] == 'User':
                st.markdown(f"**👤 You:** {entry['content']}")
            elif '[Challenge Assumptions]' in entry['speaker']:
                challenger_name = entry['speaker'].replace(' [Challenge Assumptions]', '')
                icon = MODELS_CONFIG.get(challenger_name, {}).get("icon", "🤖")
                st.markdown(f"**🕵️ {icon} {challenger_name} (Challenge Assumptions):** {entry['content']}")
            else:
                icon = MODELS_CONFIG.get(entry['speaker'], {}).get("icon", "🤖")
                st.markdown(f"**{icon} {entry['speaker']}:** {entry['content']}")
            st.markdown("---")

    if len(st.session_state.conversation_thread) > 1:
        st.markdown("### 💬 Continue the Conversation")
        follow_up = st.text_area(
            "Ask a follow-up, share more context, or redirect:",
            placeholder="Examples:\n- Can you be more specific about...\n- That's not quite right - here's what I meant...",
            height=100,
            key=f"followup_{st.session_state.follow_up_counter}"
        )
        challenge_active = st.checkbox(
            "Activate **Challenge Assumptions** mode for wider viewpoints?",
            key=f"challenge_assumptions_{st.session_state.follow_up_counter}",
            help="One model will be assigned to question the consensus to prevent groupthink."
        )
        st.markdown("""
        💡 **Pro tip:** You can steer the conversation by adding requests like:
        • "Be more direct about this" • "Focus on the core issue here" 
        • "Go deeper on that last point" • "I need more disagreement on this approach"
        """)

        if st.button("💬 Ask Follow-up", disabled=not follow_up.strip(), key=f"ask_{st.session_state.follow_up_counter}"):
            st.session_state.conversation_thread.append({
                "speaker": "User", "content": follow_up,
                "timestamp": "Follow-up", "time": datetime.now()
            })
            api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
            models = list(MODELS_CONFIG.keys())
            random.shuffle(models)

            if challenge_active:
                challenger_model = models.pop()
                other_models = models
                for model in other_models:
                    with st.spinner(f"{model} responding..."):
                        response = generate_model_response(follow_up, model, api_keys)
                        st.session_state.conversation_thread.append({
                            "speaker": model, "content": response,
                            "timestamp": "Follow-up response", "time": datetime.now()
                        })
                with st.spinner(f"{challenger_model} challenging assumptions..."):
                    response = generate_model_response(follow_up, challenger_model, api_keys, is_challenger=True)
                    st.session_state.conversation_thread.append({
                        "speaker": f"{challenger_model} [Challenge Assumptions]", "content": response,
                        "timestamp": "Challenger response", "time": datetime.now()
                    })
            else:
                for model in models:
                    with st.spinner(f"{model} responding..."):
                        response = generate_model_response(follow_up, model, api_keys)
                        st.session_state.conversation_thread.append({
                            "speaker": model, "content": response,
                            "timestamp": "Follow-up response", "time": datetime.now()
                        })
            st.session_state.follow_up_counter += 1
            st.rerun()

    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if st.session_state.session_active and len(st.session_state.conversation_thread) == 1:
    st.subheader("🎭 AI Consultants Responding")
    user_input = st.session_state.conversation_thread[0]['content']
    api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
    models = list(MODELS_CONFIG.keys())
    random.shuffle(models)

    for i, model in enumerate(models):
        with st.spinner(f"{model} thinking..."):
            response = generate_model_response(user_input, model, api_keys)
        st.session_state.conversation_thread.append({
            "speaker": model, "content": response,
            "timestamp": "Initial response", "time": datetime.now()
        })
        if i < len(models) - 1:
            time.sleep(1)
    st.rerun()

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Team of Rivals v1.8 — Clean UI + Code Quality</p>", unsafe_allow_html=True)
