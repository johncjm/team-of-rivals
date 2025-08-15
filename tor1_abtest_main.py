# tor1_abtest_main.py - A/B Test Build (2025-08-15)
# Preserves Quick Mode (single responder + reviewer), Deep Dive, and Challenge Assumptions.
# Minimal changes vs 7/27 "Pre-Beta Fix #1" to reduce regression risk:
# - Adds AB_TEST_VERSION banner and run labeling
# - Keeps API calling functions and UI structure intact
# - Retains audio transcription (Whisper), guidelines, and session state keys

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import time
import random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

AB_TEST_VERSION = "tor1_abtest_main.py — 2025-08-15"

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="Team of Rivals — A/B Test Build",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------
# Session State Init
# ----------------------
def _init_state():
    defaults = {
        "conversation_thread": [],
        "session_active": False,
        "consultation_mode": "deep",  # "quick" or "deep"
        "round_number": 1,
        "follow_up_counter": 0,
        "current_follow_up": "",
        "quick_responder": None,
        "quick_reviewer": None,
        "audio_transcription": "",
        "input_text": "",
        "follow_up_mode": None,  # used to switch mode mid-session
        "ab_label": "Condition B: API (ToR)",  # label outputs for A/B bookkeeping
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ----------------------
# API Keys
# ----------------------
try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    openai_key = st.secrets["OPENAI_API_KEY"]
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    api_available = True
except Exception as e:
    st.error(f"API keys not configured: {e}")
    api_available = False

# ----------------------
# Guidelines
# ----------------------
CONVERSATION_GUIDELINES = """
**Team of Rivals Conversation Guidelines:**

You are elite AI consultants collaborating in the world's first AI roundtable. Your reputation comes from how well the GROUP solves problems together.

**Listen & Learn First:**
• Your first job is to understand the problem. If the user gives you too little to go on, ask for more
• In many cases, your first response will be a question - that's normal and helpful
• Sometimes something as simple as "tell me more" is exactly what's needed
• If you're tempted to give general advice, that's a sign you need more specifics first
• It's better to say "I'm not sure I understand the problem yet" than to guess

**Be Human and Genuine:**
• It's fine to start with empathy: "That sounds frustrating" or "I can see why that's confusing"
• Avoid empty praise
• If you have concerns, voice them: "That could work, but I'm wondering about..."

**Conversation Flow:**
• Keep responses 2-3 paragraphs max
• Build on others' points or disagree briefly and clearly
• It's okay to say "I agree with Claude's question" and stop

**Work as a Team:**
• Challenge assumptions and point out potential problems
• You're consultants, not search engines. Understand first, then advise.

IMPORTANT: Never respond as if you are another model. Each model speaks only for themselves.
"""

REVIEWER_GUIDELINES = """
Your role is to help users assess how much they can rely on the primary response. Think of yourself as their research assistant, spotting elements that deserve caution or further verification.

**Start with a reliability emoji:**
• 🟢 reliable overall
• 🟡 useful but verify some points
• 🔴 significant issues / caution

Then give 1–2 sentences explaining your assessment, followed by concise notes on:
- Uncertain claims
- Missing context affecting conclusions
- Risky assumptions
- What to verify next
- Any contradictions

IMPORTANT: Never respond as if you are another model. You are providing your own independent review.
"""

ELEVENTH_MAN_GUIDELINES = """
**Your Role: Challenge Assumptions (The Eleventh Man)**

Provide a meaningfully different perspective to prevent groupthink.

**Do:**
• Question assumptions politely
• Surface potential risks and second-order effects
• Propose a plausible alternative
• Keep a collaborative tone

IMPORTANT: Never respond as if you are another model. You provide your own independent challenge.
"""

# ----------------------
# Utilities
# ----------------------
def transcribe_audio(audio_bytes: bytes, api_key: str) -> str:
    """Transcribe audio using OpenAI Whisper."""
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

def call_openai(prompt: str, api_key: str) -> str:
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

def call_anthropic(prompt: str, api_key: str) -> str:
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        if hasattr(response, "content") and response.content:
            content = response.content[0].text.strip()
            return content if content else "[Claude returned empty response]"
        return "[Claude returned empty response]"
    except Exception as e:
        return f"[Claude temporarily unavailable: {str(e)[:50]}...]"

def call_gemini(prompt: str, api_key: str) -> str:
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 1200}
        )
        return response.text
    except Exception as e:
        return f"[Gemini temporarily unavailable: {str(e)[:50]}...]"

def _mode_guidance(mode: str) -> str:
    if mode == "quick":
        return """
**QUICK MODE - Efficient solutions:**
• Provide clear, actionable answers
• Ask only essential clarifying questions
• Move toward practical recommendations
• Keep responses concise
"""
    return """
**DEEP DIVE MODE - Comprehensive exploration:**
• Understand full context
• Explore related problems and root causes
• Consider strategic implications
• Collaborate with other models in the discussion
"""

def _recent_context() -> str:
    if not st.session_state.conversation_thread:
        return ""
    recent: List[Dict] = []
    original = st.session_state.conversation_thread[0]
    recent.append(original)
    if len(st.session_state.conversation_thread) > 1:
        recent.extend(st.session_state.conversation_thread[-6:])
    return "\n\n".join(f"{e['speaker']}: {e['content']}" for e in recent)

def generate_model_response(
    question: str,
    model_name: str,
    api_key_dict: Dict[str, str],
    is_initial: bool = False,
    role: str = "primary",
    is_challenger: bool = False
) -> str:
    """Generate a model's response with conversation context and guidelines."""
    base_guidelines = (
        ELEVENTH_MAN_GUIDELINES if is_challenger
        else REVIEWER_GUIDELINES if role == "reviewer"
        else CONVERSATION_GUIDELINES
    )
    mode_guidance = _mode_guidance(st.session_state.consultation_mode)
    context = _recent_context()

    full_prompt = f"""{base_guidelines}

{mode_guidance}

**Run Label:** {st.session_state.ab_label}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}

You are {model_name} responding to this user query.
"""

    name = model_name.lower()
    if name in ["gpt-4", "gpt-4o", "chatgpt"]:
        return call_openai(full_prompt, api_key_dict["openai"])
    if name == "claude":
        return call_anthropic(full_prompt, api_key_dict["anthropic"])
    if name == "gemini":
        return call_gemini(full_prompt, api_key_dict["gemini"])
    return "Model not recognized"

def assign_quick_mode_roles(selected_model: str) -> Tuple[str, str]:
    """Assign responder and reviewer for Quick Mode."""
    available = ["GPT-4", "Claude", "Gemini"]
    responder = selected_model
    reviewers = [m for m in available if m != responder]
    reviewer = random.choice(reviewers) if reviewers else "Gemini"
    return responder, reviewer

def parse_reviewer_response(response: str) -> Tuple[str, str, str]:
    """Parse reviewer response for leading reliability emoji."""
    if not response or not response.strip():
        return "🔍", "Review completed", response
    response = response.strip()
    for emoji in ["🟢", "🟡", "🔴"]:
        if response.startswith(emoji):
            remaining = response[len(emoji):].strip()
            return emoji, remaining, ""
    return "🔍", response, ""

def update_input_text():
    st.session_state.input_text = st.session_state.problem_input

# ----------------------
# Header
# ----------------------
st.title("🎭 Team of Rivals — A/B Test Build")
st.caption(AB_TEST_VERSION)
st.markdown("*Let ChatGPT, Claude and Gemini collaborate — or run Quick Mode with a safety-net review.*")

# ----------------------
# Sidebar (explanatory text kept minimal)
# ----------------------
with st.sidebar:
    st.markdown("### Mode Overview")
    st.markdown("- **Quick Mode:** One model answers, another reviews for reliability.")
    st.markdown("- **Deep Dive:** Three models collaborate; optional **Challenge Assumptions**.")
    st.markdown("---")
    st.markdown("### A/B Label")
    st.markdown("Label outputs for your scoring sheet.")
    st.session_state.ab_label = st.text_input(
        "Run label (for export/logs):",
        value=st.session_state.ab_label,
        help="E.g., 'Condition B: API (ToR)'"
    )

# ----------------------
# Mode Selection
# ----------------------
st.subheader("🎯 Choose Your Consultation Style")
consultation_mode = st.radio(
    "How deep should we go?",
    [
        "🏃‍♂️ Quick & Simple — one model answers, one reviews. You can switch later.",
        "🔬 Deep Dive — all three collaborate; optional Challenge Assumptions."
    ],
    key="mode_selection"
)
st.session_state.consultation_mode = "quick" if "Quick" in consultation_mode else "deep"

# Quick Mode primary model
if st.session_state.consultation_mode == "quick":
    st.subheader("📍 Quick Mode Setup")
    selected_model = st.radio(
        "Pick the responder:",
        ["GPT-4", "Claude", "Gemini"],
        horizontal=True,
        help="A second model will automatically review the response."
    )
else:
    st.subheader("📍 Deep Dive Mode")
    st.info("All three consultants will collaborate on your question.")
    selected_model = None

# ----------------------
# Input (audio + text)
# ----------------------
st.subheader("📢 Share Your Challenge")
st.markdown("Think out loud if helpful — they’ll ask for clarification if needed.")

audio_input = st.audio_input("🎤 Record your challenge (optional)")
if audio_input is not None and api_available:
    with st.spinner("Transcribing audio..."):
        audio_bytes = audio_input.read()
        transcription = transcribe_audio(audio_bytes, openai_key)
        st.session_state.audio_transcription = transcription
        if transcription and not transcription.startswith("Error"):
            st.success("🎤 Audio transcribed.")
            with st.expander("📝 What I heard", expanded=True):
                st.write(transcription)

user_problem = st.text_area(
    "Or type your challenge here:",
    placeholder="Examples:\n- Short flawed article for edit advice\n- Quick factual verification\n- Story pitch shaping (Socratic)",
    height=120,
    key="problem_input",
    on_change=update_input_text
)

combined_input = ""
if user_problem:
    combined_input += user_problem
if st.session_state.audio_transcription and not st.session_state.audio_transcription.startswith("Error"):
    combined_input = (combined_input + "\n\n[From voice input:] " if combined_input else "") + st.session_state.audio_transcription

# ----------------------
# Start Consultation
# ----------------------
if st.button("🚀 Start AI Consultation", disabled=(not combined_input.strip() or not api_available)):
    st.session_state.conversation_thread = [{
        "speaker": "User",
        "content": combined_input,
        "timestamp": "Initial request",
        "time": datetime.now(),
        "ab_label": st.session_state.ab_label
    }]
    st.session_state.session_active = True
    st.session_state.round_number = 1
    st.rerun()

# ----------------------
# Active Session UI
# ----------------------
def _icon_for(name: str) -> str:
    if "GPT-4" in name or "ChatGPT" in name:
        return "🤖"
    if "Claude" in name:
        return "🧠"
    if "Gemini" in name:
        return "💎"
    return "🤖"

if st.session_state.session_active:
    st.markdown("---")
    st.subheader("💬 Consultation in Progress")

    # Render thread
    for entry in st.session_state.conversation_thread:
        with st.container():
            speaker = entry["speaker"]
            if speaker == "User":
                st.markdown(f"**👤 You ({entry.get('ab_label', st.session_state.ab_label)}):** {entry['content']}")
            elif speaker.endswith(" [Challenge Assumptions]"):
                base = speaker.replace(" [Challenge Assumptions]", "")
                st.markdown(f"**🕵️ {_icon_for(base)} {base} (Challenge Assumptions):** {entry['content']}")
            elif speaker.endswith(" [Reviewer]"):
                base = speaker.replace(" [Reviewer]", "")
                emoji, content, _ = parse_reviewer_response(entry["content"])
                st.markdown(f"**🔍 Review by {_icon_for(base)} {base}:** {emoji} {content}")
            else:
                st.markdown(f"**{_icon_for(speaker)} {speaker}:** {entry['content']}")
            st.markdown("---")

    # Follow-up controls
    if len(st.session_state.conversation_thread) > 1:
        st.markdown("### 💬 Continue the Conversation")

        current_mode = st.session_state.consultation_mode
        colm1, colm2 = st.columns(2)
        if current_mode == "quick":
            with colm1:
                if st.button("🏃‍♂️ Quick Follow-up", help="Same single responder + reviewer"):
                    st.session_state.follow_up_mode = "quick"
            with colm2:
                if st.button("🔬 Switch to Deep Dive", help="All three collaborate"):
                    st.session_state.follow_up_mode = "deep"
                    st.session_state.consultation_mode = "deep"
        else:
            with colm1:
                st.markdown("")  # spacer
            with colm2:
                if st.button("🏃‍♂️ Switch to Quick Mode", help="Faster single answer + review"):
                    st.session_state.follow_up_mode = "quick"
                    st.session_state.consultation_mode = "quick"

        follow_up = st.text_area(
            "Ask a follow-up, share more context, or redirect:",
            placeholder="e.g., 'Be specific about X', 'That's not quite right — I meant Y', 'Go deeper on Z'",
            height=80,
            key=f"followup_{st.session_state.follow_up_counter}",
            value=""
        )

        challenge_active = False
        if st.session_state.consultation_mode == "deep":
            challenge_active = st.checkbox(
                "Activate **Challenge Assumptions**?",
                key=f"challenge_assumptions_{st.session_state.follow_up_counter}",
                help="Assign one model to question consensus and surface risks."
            )

        st.markdown("""
        💡 Steering hints: "Be more direct" • "Focus on the core issue" • "Go deeper on that last point" • "I need more disagreement"
        """)

        if st.button("💬 Ask Follow-up", disabled=not follow_up.strip(), key=f"ask_{st.session_state.follow_up_counter}"):
            st.session_state.conversation_thread.append({
                "speaker": "User",
                "content": follow_up,
                "timestamp": "Follow-up question",
                "time": datetime.now(),
                "ab_label": st.session_state.ab_label
            })

            api_keys = {"openai": openai_key, "anthropic": anthropic_key, "gemini": gemini_key}
            intended_mode = st.session_state.get("follow_up_mode", st.session_state.consultation_mode)

            if intended_mode == "quick":
                # Assign persistent roles within session
                if not st.session_state.quick_responder:
                    responder, reviewer = assign_quick_mode_roles(selected_model or "GPT-4")
                    st.session_state.quick_responder = responder
                    st.session_state.quick_reviewer = reviewer
                else:
                    responder = st.session_state.quick_responder
                    reviewer = st.session_state.quick_reviewer

                # Primary
                with st.spinner(f"{responder} responding..."):
                    primary = generate_model_response(follow_up, responder, api_keys, is_initial=False)
                    st.session_state.conversation_thread.append({
                        "speaker": responder,
                        "content": primary,
                        "timestamp": "Follow-up response",
                        "time": datetime.now()
                    })

                # Review
                with st.spinner(f"{reviewer} reviewing..."):
                    review_prompt = f"Review this response for reliability: {primary}"
                    review = generate_model_response(review_prompt, reviewer, api_keys, is_initial=False, role="reviewer")
                    st.session_state.conversation_thread.append({
                        "speaker": f"{reviewer} [Reviewer]",
                        "content": review,
                        "timestamp": "Follow-up review",
                        "time": datetime.now()
                    })

            else:
                # Deep Dive
                models = ["GPT-4", "Claude", "Gemini"]
                random.shuffle(models)

                if challenge_active:
                    challenger = models.pop()
                    for m in models:
                        with st.spinner(f"{m} responding..."):
                            resp = generate_model_response(follow_up, m, api_keys, is_initial=False)
                            st.session_state.conversation_thread.append({
                                "speaker": m,
                                "content": resp,
                                "timestamp": "Follow-up response",
                                "time": datetime.now()
                            })
                    with st.spinner(f"{challenger} challenging assumptions..."):
                        chal = generate_model_response(follow_up, challenger, api_keys, is_initial=False, is_challenger=True)
                        st.session_state.conversation_thread.append({
                            "speaker": f"{challenger} [Challenge Assumptions]",
                            "content": chal,
                            "timestamp": "Challenger response",
                            "time": datetime.now()
                        })
                else:
                    for m in models:
                        with st.spinner(f"{m} responding..."):
                            resp = generate_model_response(follow_up, m, api_keys, is_initial=False)
                            st.session_state.conversation_thread.append({
                                "speaker": m,
                                "content": resp,
                                "timestamp": "Follow-up response",
                                "time": datetime.now()
                            })

            st.session_state.follow_up_counter += 1
            st.session_state.follow_up_mode = None
            st.rerun()

    # Reset
    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        for key in [
            "conversation_thread", "session_active", "round_number", "follow_up_counter",
            "current_follow_up", "quick_responder", "quick_reviewer", "follow_up_mode",
            "audio_transcription", "input_text"
        ]:
            if key in st.session_state:
                del st.session_state[key]
        _init_state()
        st.rerun()

# ----------------------
# Initial Responses
# ----------------------
if st.session_state.session_active and len(st.session_state.conversation_thread) == 1:
    st.markdown("---")
    st.subheader("🎭 AI Consultants Responding")

    user_input = st.session_state.conversation_thread[0]["content"]
    api_keys = {"openai": openai_key, "anthropic": anthropic_key, "gemini": gemini_key}

    if st.session_state.consultation_mode == "quick":
        responder, reviewer = assign_quick_mode_roles(selected_model or "GPT-4")
        st.session_state.quick_responder = responder
        st.session_state.quick_reviewer = reviewer

        with st.spinner(f"{responder} thinking..."):
            primary = generate_model_response(user_input, responder, api_keys, is_initial=True)
        st.session_state.conversation_thread.append({
            "speaker": responder,
            "content": primary,
            "timestamp": "Initial response",
            "time": datetime.now()
        })
        st.markdown(f"### {_icon_for(responder)} {responder}")
        st.markdown(primary)

        with st.spinner(f"{reviewer} reviewing..."):
            review_prompt = f"Review this response for reliability: {primary}"
            review = generate_model_response(review_prompt, reviewer, api_keys, is_initial=True, role="reviewer")
        st.session_state.conversation_thread.append({
            "speaker": f"{reviewer} [Reviewer]",
            "content": review,
            "timestamp": "Initial review",
            "time": datetime.now()
        })
        emoji, content, _ = parse_reviewer_response(review)
        st.markdown(f"### 🔍 Review by {_icon_for(reviewer)} {reviewer}")
        st.markdown(f"{emoji} {content}")

    else:
        models = ["GPT-4", "Claude", "Gemini"]
        random.shuffle(models)
        for i, m in enumerate(models):
            with st.spinner(f"{m} thinking..."):
                resp = generate_model_response(user_input, m, api_keys, is_initial=True)
            st.session_state.conversation_thread.append({
                "speaker": m,
                "content": resp,
                "timestamp": "Initial response",
                "time": datetime.now()
            })
            st.markdown(f"### {_icon_for(m)} {m}")
            st.markdown(resp)
            if i < len(models) - 1:
                time.sleep(1)

    st.rerun()

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption(f"{AB_TEST_VERSION} • Preserves Quick Mode, Deep Dive, Challenge Assumptions, audio, and API calls.")
