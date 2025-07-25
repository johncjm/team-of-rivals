# tor1_main.py - Pre-Beta Version with UI Fixes
# Team of Rivals: Complete implementation with Challenge Assumptions + UI improvements

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import re
import json
import time
import random
from typing import Dict, List, Optional
from datetime import datetime

# Configure page with wider sidebar
st.set_page_config(
    page_title="Team of Rivals", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for wider sidebar - Updated for Streamlit 1.47.0
st.markdown("""
<style>
    /* Wider sidebar for better content display */
    .css-1d391kg, .css-1cypcdb, .st-emotion-cache-16txtl3, .eczjsme0 {
        width: 400px !important;
        max-width: 400px !important;
    }
    
    /* Alternative approach - target sidebar container */
    section[data-testid="stSidebar"] > div {
        width: 400px !important;
        max-width: 400px !important;
    }
    
    /* Ensure main content adjusts */
    .main .block-container {
        max-width: calc(100% - 420px) !important;
        margin-left: 420px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation_thread' not in st.session_state:
    st.session_state.conversation_thread = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'consultation_mode' not in st.session_state:
    st.session_state.consultation_mode = "deep"  # "quick" or "deep"
if 'round_number' not in st.session_state:
    st.session_state.round_number = 1
if 'follow_up_counter' not in st.session_state:
    st.session_state.follow_up_counter = 0
if 'current_follow_up' not in st.session_state:
    st.session_state.current_follow_up = ""
if 'quick_responder' not in st.session_state:
    st.session_state.quick_responder = None
if 'quick_reviewer' not in st.session_state:
    st.session_state.quick_reviewer = None
if 'audio_transcription' not in st.session_state:
    st.session_state.audio_transcription = ""
# UI state for button enabling
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

# Load API keys
try:
    gemini_key = st.secrets["GOOGLE_API_KEY"]
    openai_key = st.secrets["OPENAI_API_KEY"] 
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    api_available = True
except Exception as e:
    st.error(f"API keys not configured: {e}")
    api_available = False

# Enhanced conversation guidelines from v15
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
• Genuine encouragement is valuable: "That's an interesting approach" or "You're thinking about this the right way"
• But avoid empty praise - "That's brilliant!" when you don't know enough yet feels fake
• If you have concerns, voice them: "That could work, but I'm wondering about..." is more helpful than false enthusiasm

**Conversation Flow:**
• This is a conversation, not a conference presentation - keep responses 2-3 paragraphs max
• Build on others' points: "Building on Claude's insight..." or "I disagree with GPT-4 because..."
• Add your perspective only if it's genuinely different or builds meaningfully on what's been said
• It's okay to say "I agree with Claude's question" and leave it there
• Sometimes the most helpful response is stating your understanding: "So it sounds like..."

**Work as a Team:**
• Challenge assumptions and point out potential problems - productive disagreement helps users
• It's okay to say "I'm not sure" or "Let me think about this differently"
• Your expertise is most valuable when applied to the actual situation, not hypothetical ones
• You're consultants, not search engines. Consultants understand first, then advise.

Remember: You're not alone - your collaborators will step up and challenge you if needed.

IMPORTANT: Never respond as if you are another model. Each model speaks only for themselves in collaborative discussions.
"""

# Quick Mode Reviewer Guidelines
REVIEWER_GUIDELINES = """
Your role is to help users assess how much they can rely on the primary response. Think of yourself as their research assistant, spotting elements that deserve caution or further verification.

**Start your response with a reliability assessment emoji:**
• 🟢 if the information looks reliable and you don't see significant concerns
• 🟡 if it's useful but worth verifying some points or adding context
• 🔴 if there are significant issues or the user should approach with caution

**After the emoji, explain your assessment in 1-2 sentences. Examples:**

🟢 This covers the key points well - the approaches mentioned are standard best practices, and the specific suggestions are appropriate for this type of project.

🟡 Good framework, but this doesn't address the transition process for current visiting students - you'll want to contact admissions about simplified procedures for your specific situation.

🔴 Several specific claims here need verification. The 2025 salary projections don't specify data sources, and the market predictions assume stable economic conditions.

**What to look for:**
- Claims that seem uncertain or hard to verify
- Missing context that could change conclusions  
- Assumptions that might not hold in all cases
- Areas where the user should do additional research
- Information that contradicts what you know (share your perspective for comparison)

Your goal is to put the user in a better position to assess the response. Focus on being helpful - whether that's flagging concerns or confirming the response looks solid.

IMPORTANT: Never respond as if you are another model. You are providing your own independent review.
"""

# Challenge Assumptions (Eleventh Man) Guidelines
ELEVENTH_MAN_GUIDELINES = """
**Your Role: Challenge Assumptions (The Eleventh Man)**

Your designated role for this round is to be the "Eleventh Man." Even if you lean toward agreeing with the user or the other models, your duty is to constructively challenge the consensus to prevent groupthink.

**Your Goal:** To strengthen the final outcome by stress-testing the ideas on the table.

**You must contribute a meaningfully different perspective** — even if you ultimately agree, your job is to explore what we may have missed.

**How to Behave:**
• **Question Assumptions:** What unstated beliefs is the current approach built on? Politely ask about them. ("This approach seems to assume X, have we considered Y?")
• **Surface Potential Risks:** What could go wrong with the proposed plan? What are the second-order consequences?
• **Propose a Plausible Alternative:** Construct the best possible argument for a different path. This isn't about being difficult; it's about ensuring all reasonable options are explored.
• **Maintain a Collaborative Tone:** You are a "critical friend," not an adversary. Your goal is to help, not to win an argument. Frame your points constructively.

Remember: You are a vital part of the team. Your job is to ensure that the group's conclusion is as robust and well-vetted as possible.

IMPORTANT: Never respond as if you are another model. You are providing your own independent challenge to the group's thinking.
"""

# Audio transcription function (restored from v15)
def transcribe_audio(audio_bytes, api_key):
    """Transcribe audio using OpenAI Whisper"""
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

# API calling functions
def call_openai(prompt, api_key):
    """Call OpenAI GPT-4 API"""
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
    """Call Anthropic Claude API with enhanced handling"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if hasattr(response, 'content') and response.content:
            content = response.content[0].text.strip()
            if content:
                return content
            else:
                return "[Claude returned empty response]"
        else:
            return "[Claude returned empty response]"
            
    except Exception as e:
        return f"[Claude temporarily unavailable: {str(e)[:50]}...]"

def call_gemini(prompt, api_key):
    """Call Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 1200,
            }
        )
        return response.text
    except Exception as e:
        return f"[Gemini temporarily unavailable: {str(e)[:50]}...]"

def generate_model_response(question: str, model_name: str, api_key_dict: Dict, 
                           is_initial: bool = False, role: str = "primary", 
                           is_challenger: bool = False) -> str:
    """Generate a model's response with conversation context and guidelines"""
    
    # Get recent context
    recent_thread = []
    if st.session_state.conversation_thread:
        original_entry = st.session_state.conversation_thread[0]
        recent_thread.append(original_entry)
        
        if len(st.session_state.conversation_thread) > 1:
            recent_entries = st.session_state.conversation_thread[-6:]
            recent_thread.extend(recent_entries)
    
    # Build context string
    context = "\n\n".join([
        f"{entry['speaker']}: {entry['content']}" 
        for entry in recent_thread
    ])
    
    # Select appropriate guidelines based on role
    if is_challenger:
        base_guidelines = ELEVENTH_MAN_GUIDELINES
    elif role == "reviewer":
        base_guidelines = REVIEWER_GUIDELINES
    else:
        base_guidelines = CONVERSATION_GUIDELINES
    
    # Mode-specific guidance
    if st.session_state.consultation_mode == "quick":
        mode_guidance = """
**QUICK MODE - User wants efficient solutions:**
• Focus on providing clear, actionable answers
• Ask only essential clarifying questions
• Move toward practical recommendations
• Keep responses focused and concise
"""
    else:  # deep mode
        mode_guidance = """
**DEEP DIVE MODE - User wants comprehensive exploration:**
• Take time to understand the full context
• Explore related problems and root causes
• Consider strategic implications and systemic factors
• Collaborate naturally with other models in the discussion
"""
    
    # Create full prompt
    full_prompt = f"""{base_guidelines}

{mode_guidance}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}

You are {model_name} responding to this user query."""

    # Call appropriate API
    if model_name.lower() in ["gpt-4", "gpt-4o"]:
        return call_openai(full_prompt, api_key_dict['openai'])
    elif model_name.lower() == "claude":
        return call_anthropic(full_prompt, api_key_dict['anthropic'])
    elif model_name.lower() == "gemini":
        return call_gemini(full_prompt, api_key_dict['gemini'])
    else:
        return "Model not recognized"

def assign_quick_mode_roles(selected_model):
    """Assign responder and reviewer for Quick Mode"""
    available_models = ["GPT-4", "Claude", "Gemini"]
    responder = selected_model
    
    # Choose reviewer from remaining models
    available_reviewers = [m for m in available_models if m != responder]
    reviewer = random.choice(available_reviewers) if available_reviewers else "Gemini"
    
    return responder, reviewer

def parse_reviewer_response(response: str) -> tuple:
    """Parse reviewer response for emoji and content"""
    if not response or not response.strip():
        return "🔍", "Review completed", response
    
    response = response.strip()
    
    # Check for reliability emojis at the start
    reliability_emojis = ["🟢", "🟡", "🔴"]
    
    for emoji in reliability_emojis:
        if response.startswith(emoji):
            # Remove emoji and get remaining text
            remaining = response[len(emoji):].strip()
            return emoji, remaining, ""
    
    # Fallback - return whole response
    return "🔍", response, ""

# Callback function for enabling button
def update_input_text():
    st.session_state.input_text = st.session_state.problem_input
    # PART 2: UI and Main Application Logic
# This goes directly after Part 1

# Main UI with new text and layout
st.title("🎭 Team of Rivals")
st.markdown("*Let ChatGPT, Claude and Gemini collaborate to answer your questions or dig into your toughest challenges*")

# Sidebar with comprehensive explanation (wider)
with st.sidebar:
    st.markdown("### How Team of Rivals Works")
    st.markdown("""
    Are four heads (yours plus 3 LLMs) better than one? We think so, and we've built this app to try to find out!

    Are you tired of asking an AI bot a simple question and not being sure how much to trust the answer? In Quick Mode, one model gives an answer and another model checks it out for you.

    In Deep Dive Mode, all three models work together, first in a Listen & Learn phase to scope out your issue, whether it's a coding question or understanding why your screenplay isn't coming together. Then they switch to Build, Execute & Check, pooling their ideas and challenging each other's thinking to come up with your best path forward.

    Could you do the same with a lot of copy and pasting? Yes, kinda! But besides cutting way down on that work, Team of Rivals includes prompt engineering developed in collaboration with the models themselves to mitigate their quirks and maximize their usefulness. Hallucinations? Sycophancy? We can't guarantee they'll go away, but we think you're on a better path than working with a bot in the wild.

    As you work, you'll get suggestions for how to steer the discussion in the direction you want. And for a Deep Dive, you can activate "Challenge Assumptions" mode, in which one model is asked to be the fresh eye questioning assumptions and looking for weaknesses in proposed solutions.

    **For the best results, be an active participant, not a passive listener:**

    • **Speak freely** - ramble, think out loud, contradict yourself, especially if using audio for the initial input. That's how real insights emerge.
    • **Push back** if the discussion seems off track or missing something important.
    • **Ask follow-ups** when you want them to dig deeper or explore different angles.
    • **Don't worry about structure** - the consultants will help organize and focus the discussion.

    **Remember:** The best consultations happen when you stay engaged and guide the conversation toward what you actually need!
    """)

st.markdown("---")

# Consultation mode selection
st.subheader("🎯 Choose Your Consultation Style")
consultation_mode = st.radio(
    "How deep should we go?",
    ["🏃‍♂️ Quick & Simple — one model, one answer plus one review. Can switch to Deep mode later.",
     "🔬 Deep Dive — work with all three models to dig into your challenge and collaborate on solutions"],
    key="mode_selection"
)

st.session_state.consultation_mode = "quick" if "Quick" in consultation_mode else "deep"
# DEBUG SECTION - Remove after fixing
with st.expander("🔧 Debug Info (remove after testing)"):
    st.write("**Streamlit Version:**", st.__version__)
    st.write("**Session State Keys:**", list(st.session_state.keys()))
    st.write("**Consultation Mode:**", st.session_state.consultation_mode)
    st.write("**Session Active:**", st.session_state.session_active)
    st.write("**Audio Transcription:**", st.session_state.audio_transcription[:100] + "..." if len(st.session_state.audio_transcription) > 100 else st.session_state.audio_transcription)
    
    # Test the CSS
    st.markdown("**CSS Test:** This sidebar should be 400px wide")
    
    # Test button state
    if 'user_problem' in locals():
        st.write("**Current problem input:**", repr(user_problem))
    else:
        st.write("**Current problem input:** Not defined yet")
    
st.markdown("---")

# Model selection for Quick Mode
if st.session_state.consultation_mode == "quick":
    st.subheader("📍 Quick Mode Setup")
    selected_model = st.radio(
        "Pick one model to answer your question:",
        ["GPT-4", "Claude", "Gemini"],
        horizontal=True,
        help="A second model will automatically review the response for reliability."
    )
else:
    st.subheader("📍 Deep Dive Mode")
    st.info("All three consultants will collaborate on your question.")
    selected_model = None

# Problem input with audio (restored from v15)
st.subheader("📢 Share Your Challenge")
st.markdown("**Think out loud:** Complex problems often need rambling to understand properly. Don't worry about being perfectly clear - the consultants will ask follow-up questions!")

# Audio input
audio_input = st.audio_input("🎤 Record your challenge (captures nuance and context)")
if audio_input is not None and api_available:
    with st.spinner("Transcribing your audio..."):
        audio_bytes = audio_input.read()
        transcription = transcribe_audio(audio_bytes, openai_key)
        st.session_state.audio_transcription = transcription
        
        if transcription and not transcription.startswith("Error"):
            st.success("🎤 Audio transcribed!")
            with st.expander("📝 What I heard", expanded=True):
                st.write(transcription)

# Text input with auto-enabling callback
user_problem = st.text_area(
    "Or type your challenge here:",
    placeholder="Examples:\n- I have some Python code that feels messy\n- Need help planning a difficult team conversation\n- Trying to decide between two strategic directions\n- Working on a presentation that isn't clicking\n\nDon't worry about being perfectly clear - they'll ask questions!",
    height=120,
    key="problem_input",
    on_change=update_input_text
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

# Start consultation button (auto-enabled)
button_disabled = not combined_input.strip() or not api_available
if st.button("🚀 Start AI Consultation", disabled=button_disabled):
    
    # Initialize conversation thread
    st.session_state.conversation_thread = [{
        "speaker": "User",
        "content": combined_input,
        "timestamp": "Initial request",
        "time": datetime.now()
    }]
    
    st.session_state.session_active = True
    st.session_state.round_number = 1
    st.rerun()

# Show active consultation interface
if st.session_state.session_active:
    
    # Display conversation
    st.markdown("---")
    st.subheader("💬 Consultation in Progress")
    
    # Show conversation thread (preserved throughout)
    for entry in st.session_state.conversation_thread:
        with st.container():
            if entry['speaker'] == 'User':
                st.markdown(f"**👤 You:** {entry['content']}")
            elif entry['speaker'].endswith(' [Challenge Assumptions]'):
                # Special formatting for challenger
                challenger_name = entry['speaker'].replace(' [Challenge Assumptions]', '')
                icon = "🤖" if "GPT-4" in challenger_name else "🧠" if "Claude" in challenger_name else "💎"
                st.markdown(f"**🕵️ {icon} {challenger_name} (Challenge Assumptions):** {entry['content']}")
            elif entry['speaker'].endswith(' [Reviewer]'):
                # Show reviewer responses with emoji parsing
                reviewer_name = entry['speaker'].replace(' [Reviewer]', '')
                icon = "🤖" if "GPT-4" in reviewer_name else "🧠" if "Claude" in reviewer_name else "💎"
                
                emoji, content, explanation = parse_reviewer_response(entry['content'])
                st.markdown(f"**🔍 Review by {icon} {reviewer_name}:** {emoji} {content}")
            else:
                icon = "🤖" if "GPT-4" in entry['speaker'] else "🧠" if "Claude" in entry['speaker'] else "💎"
                st.markdown(f"**{icon} {entry['speaker']}:** {entry['content']}")
            st.markdown("---")
    
    # Continue conversation section (IMPROVED LAYOUT)
    if len(st.session_state.conversation_thread) > 1:  # After initial responses
        st.markdown("### 💬 Continue the Conversation")
        
        # Mode-specific continuation options
        current_mode = st.session_state.consultation_mode
        
        if current_mode == "quick":
            st.markdown("Continue in Quick Mode, or ready to go Deep?")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏃‍♂️ Quick Follow-up", help="Same model + reviewer approach"):
                    st.session_state.follow_up_mode = "quick"
            with col2:
                if st.button("🔬 Switch to Deep Dive", help="All consultants collaborate"):
                    st.session_state.follow_up_mode = "deep"
                    st.session_state.consultation_mode = "deep"
        else:
            st.markdown("**Deep Dive Mode:** All consultants collaborate")
        
        st.markdown("---")
        
        # Follow-up input (REORDERED - now before Challenge Assumptions)
        follow_up = st.text_area(
            "Ask a follow-up, share more context, or redirect:",
            placeholder="Examples:\n- Can you be more specific about...\n- That's not quite right - here's what I meant...\n- I like that direction, but what about...\n- You're overcomplicating this - I just need...",
            height=80,
            key=f"followup_{st.session_state.follow_up_counter}",
            value=""
        )
        
        # Challenge Assumptions checkbox (now below input)
        challenge_active = False
        if st.session_state.consultation_mode == "deep":
            challenge_active = st.checkbox(
                "Activate **Challenge Assumptions** mode for wider viewpoints?",
                key=f"challenge_assumptions_{st.session_state.follow_up_counter}",
                help="Uses the 'Eleventh Man' approach - one model questions consensus to prevent groupthink"
            )
        
        # Pro tip for steering (stays below)
        st.markdown("""
        💡 **Pro tip:** You can steer the conversation by adding requests like:
        • "Be more direct about this" • "Focus on the core issue here" 
        • "Go deeper on that last point" • "I need more disagreement on this approach"
        """)
        
        # Follow-up processing logic
        follow_up_button_disabled = not follow_up.strip()
        if st.button("💬 Ask Follow-up", disabled=follow_up_button_disabled, key=f"ask_{st.session_state.follow_up_counter}"):
            
            # Add user question to thread
            st.session_state.conversation_thread.append({
                "speaker": "User",
                "content": follow_up,
                "timestamp": "Follow-up question",
                "time": datetime.now()
            })
            
            api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
            
            # Determine intended mode
            intended_mode = st.session_state.get("follow_up_mode", current_mode)
            
            if intended_mode == "quick":
                # Quick Mode: use assigned responder and reviewer
                if not st.session_state.quick_responder:
                    responder, reviewer = assign_quick_mode_roles(selected_model or "GPT-4")
                    st.session_state.quick_responder = responder
                    st.session_state.quick_reviewer = reviewer
                else:
                    responder = st.session_state.quick_responder
                    reviewer = st.session_state.quick_reviewer
                
                # Get primary response
                with st.spinner(f"{responder} responding..."):
                    response = generate_model_response(follow_up, responder, api_keys, is_initial=False)
                    st.session_state.conversation_thread.append({
                        "speaker": responder,
                        "content": response,
                        "timestamp": "Follow-up response",
                        "time": datetime.now()
                    })
                
                # Get reviewer response
                with st.spinner(f"{reviewer} reviewing..."):
                    review_prompt = f"Review this response for reliability: {response}"
                    review_response = generate_model_response(review_prompt, reviewer, api_keys, is_initial=False, role="reviewer")
                    st.session_state.conversation_thread.append({
                        "speaker": f"{reviewer} [Reviewer]",
                        "content": review_response,
                        "timestamp": "Follow-up review",
                        "time": datetime.now()
                    })
            
            else:
                # Deep Dive Mode with optional Challenge Assumptions
                models = ["GPT-4", "Claude", "Gemini"]
                random.shuffle(models)
                
                if challenge_active:
                    # Challenge Assumptions mode - one model becomes challenger
                    challenger_model = models.pop()  # Random selection
                    other_models = models
                    
                    # Standard collaborators respond first
                    for model in other_models:
                        with st.spinner(f"{model} responding..."):
                            response = generate_model_response(follow_up, model, api_keys, is_initial=False)
                            st.session_state.conversation_thread.append({
                                "speaker": model,
                                "content": response,
                                "timestamp": "Follow-up response",
                                "time": datetime.now()
                            })
                    
                    # Challenger responds last
                    with st.spinner(f"{challenger_model} challenging assumptions..."):
                        response = generate_model_response(follow_up, challenger_model, api_keys, is_initial=False, is_challenger=True)
                        st.session_state.conversation_thread.append({
                            "speaker": f"{challenger_model} [Challenge Assumptions]",
                            "content": response,
                            "timestamp": "Challenger response",
                            "time": datetime.now()
                        })
                
                else:
                    # Standard Deep Dive - all models respond normally
                    for model in models:
                        with st.spinner(f"{model} responding..."):
                            response = generate_model_response(follow_up, model, api_keys, is_initial=False)
                            st.session_state.conversation_thread.append({
                                "speaker": model,
                                "content": response,
                                "timestamp": "Follow-up response",
                                "time": datetime.now()
                            })
            
            # Clear input and reset state (fixed counter approach)
            st.session_state.follow_up_counter += 1
            st.session_state.follow_up_mode = None  # Reset mode selection
            st.rerun()
    
    # Session management (MOVED TO BOTTOM)
    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        # Reset session
        for key in ['conversation_thread', 'session_active', 'round_number', 'follow_up_counter', 'current_follow_up', 'quick_responder', 'quick_reviewer', 'follow_up_mode', 'audio_transcription', 'input_text']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Initial responses (direct execution style)
if st.session_state.session_active and len(st.session_state.conversation_thread) == 1:
    st.markdown("---")
    st.subheader("🎭 AI Consultants Responding")
    
    user_input = st.session_state.conversation_thread[0]['content']
    api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
    
    if st.session_state.consultation_mode == "quick":
        # Quick Mode: Single responder + reviewer
        responder, reviewer = assign_quick_mode_roles(selected_model)
        st.session_state.quick_responder = responder
        st.session_state.quick_reviewer = reviewer
        
        # Primary response
        with st.spinner(f"{responder} thinking..."):
            response = generate_model_response(user_input, responder, api_keys, is_initial=True)
        
        st.session_state.conversation_thread.append({
            "speaker": responder,
            "content": response,
            "timestamp": "Initial response",
            "time": datetime.now()
        })
        
        icon = "🤖" if responder == "GPT-4" else "🧠" if responder == "Claude" else "💎"
        st.markdown(f"### {icon} {responder}")
        st.markdown(response)
        
        # Reviewer response
        with st.spinner(f"{reviewer} reviewing..."):
            review_prompt = f"Review this response for reliability: {response}"
            review_response = generate_model_response(review_prompt, reviewer, api_keys, is_initial=True, role="reviewer")
        
        st.session_state.conversation_thread.append({
            "speaker": f"{reviewer} [Reviewer]",
            "content": review_response,
            "timestamp": "Initial review",
            "time": datetime.now()
        })
        
        # Display review with proper emoji parsing
        reviewer_icon = "🤖" if reviewer == "GPT-4" else "🧠" if reviewer == "Claude" else "💎"
        emoji, content, explanation = parse_reviewer_response(review_response)
        st.markdown(f"### 🔍 Review by {reviewer_icon} {reviewer}")
        st.markdown(f"{emoji} {content}")
    
    else:
        # Deep Dive Mode: All models respond
        models = ["GPT-4", "Claude", "Gemini"]
        random.shuffle(models)
        
        for i, model in enumerate(models):
            with st.spinner(f"{model} thinking..."):
                response = generate_model_response(user_input, model, api_keys, is_initial=True)
            
            st.session_state.conversation_thread.append({
                "speaker": model,
                "content": response,
                "timestamp": "Initial response",
                "time": datetime.now()
            })
            
            icon = "🤖" if model == "GPT-4" else "🧠" if model == "Claude" else "💎"
            st.markdown(f"### {icon} {model}")
            st.markdown(response)
            
            # Small delay between responses
            if i < len(models) - 1:
                time.sleep(1)
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Team of Rivals** - Where AI consultants collaborate to solve your toughest challenges 🎭✨")
