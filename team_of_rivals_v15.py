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

# Configure page
st.set_page_config(
    page_title="Team of Rivals", 
    page_icon="🎭",
    layout="wide"
)

# Initialize session state
if 'conversation_thread' not in st.session_state:
    st.session_state.conversation_thread = []
if 'generated_questions' not in st.session_state:
    st.session_state.generated_questions = []
if 'session_active' not in st.session_state:
    st.session_state.session_active = False
if 'audio_transcription' not in st.session_state:
    st.session_state.audio_transcription = ""
if 'consultation_mode' not in st.session_state:
    st.session_state.consultation_mode = "deep"
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = "listen_learn"
if 'session_start_time' not in st.session_state:
    st.session_state.session_start_time = None
if 'last_checkin_time' not in st.session_state:
    st.session_state.last_checkin_time = None
if 'follow_up_counter' not in st.session_state:
    st.session_state.follow_up_counter = 0
if 'current_follow_up' not in st.session_state:
    st.session_state.current_follow_up = ""
if 'initial_responses_complete' not in st.session_state:
    st.session_state.initial_responses_complete = False

# Load API keys
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    openai_key = st.secrets["OPENAI_API_KEY"] 
    anthropic_key = st.secrets["ANTHROPIC_API_KEY"]
    api_available = True
except Exception as e:
    st.error(f"API keys not configured: {e}")
    api_available = False

# Enhanced conversation guidelines
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

**Credibility Management:**
• Credibility is like tokens - spend them all and the session stops
• Never call everything "brilliant" or "excellent" - give honest assessments
• Uncertainty is strength when you have collaborators to catch errors

Remember: You're not alone - your collaborators will step up and challenge you if needed.
"""

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

def call_openai(prompt, api_key):
    """Call OpenAI GPT-4 API"""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,  # Further reduced for conciseness
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling GPT-4: {str(e)}"

def call_anthropic(prompt, api_key):
    """Call Anthropic Claude API"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1200,  # Further reduced for conciseness
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Error calling Claude: {str(e)}"

def call_gemini(prompt, api_key):
    """Call Google Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 1200,  # Further reduced for conciseness
            }
        )
        return response.text
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

def generate_model_response(question: str, model_name: str, api_key_dict: Dict, is_initial: bool = False) -> str:
    """Generate a model's response with conversation context and guidelines"""
    
    # Get recent context (original question + last 6 entries)
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
    
    # Mode-specific guidance
    if st.session_state.consultation_mode == "quick":
        mode_guidance = """
**QUICK MODE - User wants to get to solutions efficiently:**
• Restrict Listen & Learn to only what's essential for providing a good answer
• Ask focused questions if clarification is needed, but keep them minimal  
• Move toward actionable recommendations more quickly
• Avoid extensive exploration of tangential issues
• If you fully agree with a previous response, a simple "Agreed - that's the right approach" with any implementation details is sufficient
"""
    else:  # deep mode
        mode_guidance = """
**DEEP DIVE MODE - User wants comprehensive exploration:**
• Keep the solution in mind but think about the broader issues being raised
• Explore related problems and root causes that might not be immediately obvious
• Consider strategic implications and systemic factors
• Take time to understand the full context before recommending solutions
"""
    
    # Collaborative efficiency guidance (for both modes)
    collaborative_guidance = """
**Collaborative Efficiency:**
• If a previous response already covers the key points well, it's perfectly fine to say "I agree with [model]'s analysis" or "That captures it well" and add only genuinely new insights
• Brief supportive responses like "Claude's right about X, and I'd just add..." are valuable
• Quality over quantity - a short, targeted response is better than repeating what's already been said
"""
    
    # Special handling for initial responses to encourage better conversation starters
    if is_initial:
        phase_guidance = """
**IMPORTANT - This is your first response to the user's request:**
• Focus on understanding the problem before offering solutions
• If the request is vague, ask for clarification rather than giving generic advice
• It's often better to ask "What specifically is causing problems?" than to list general principles
• Start with empathy if appropriate: "That sounds challenging" or "I can see why that's frustrating"
• Remember: your expertise is most valuable when applied to their actual situation, not hypothetical ones
"""
    else:
        phase_guidance = ""
    
    # Create full prompt with guidelines
    full_prompt = f"""{CONVERSATION_GUIDELINES}

{mode_guidance}

{collaborative_guidance}

{phase_guidance}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}

You are {model_name} participating as yourself in this AI consulting roundtable. Respond in the collaborative style described above."""

    if model_name.lower() == "gpt-4":
        return call_openai(full_prompt, api_key_dict['openai'])
    elif model_name.lower() == "claude":
        return call_anthropic(full_prompt, api_key_dict['anthropic'])
    elif model_name.lower() == "gemini":
        return call_gemini(full_prompt, api_key_dict['gemini'])
    else:
        return "Model not recognized"

def check_for_gemini_moderation():
    """Check if Gemini should do a 5-minute check-in"""
    if not st.session_state.session_start_time:
        return False
    
    current_time = time.time()
    elapsed = current_time - st.session_state.session_start_time
    
    # Check every 5 minutes (300 seconds)
    if st.session_state.last_checkin_time is None:
        should_checkin = elapsed >= 300
    else:
        should_checkin = (current_time - st.session_state.last_checkin_time) >= 300
    
    return should_checkin

def gemini_checkin(api_key_dict: Dict) -> str:
    """Generate Gemini's 5-minute pulse check"""
    checkin_prompt = f"""{CONVERSATION_GUIDELINES}

You are Gemini serving as conversation moderator. It's been about 5 minutes of active discussion. 

Based on the conversation so far, provide a brief pulse check:
- Are we still exploring and understanding the problem?
- Are we ready to shift toward solutions and recommendations?
- Should we continue the current direction?

Keep this brief - just a quick temperature check to help guide the conversation forward.

Recent conversation context:
{get_recent_context_string()}
"""
    
    return call_gemini(checkin_prompt, api_key_dict['gemini'])

def get_recent_context_string() -> str:
    """Get recent context as a string"""
    if len(st.session_state.conversation_thread) <= 6:
        relevant_entries = st.session_state.conversation_thread
    else:
        relevant_entries = [st.session_state.conversation_thread[0]] + st.session_state.conversation_thread[-6:]
    
    return "\n\n".join([
        f"{entry['speaker']}: {entry['content']}" 
        for entry in relevant_entries
    ])

def generate_consensus_document(api_key_dict: Dict) -> str:
    """Generate final consensus document"""
    context = get_recent_context_string()
    
    consensus_prompt = f"""Based on this AI consulting session, create a comprehensive summary document.

**Start with 1-2 paragraphs of context** for readers unfamiliar with the session, explaining:
- What problem the user brought to the consultation
- How the discussion evolved and what insights emerged
- The collaborative process that led to the recommendations

**Then include:**
- Key insights and recommendations
- Areas of agreement among the consultants
- Areas of disagreement or alternative approaches (if any)
- Next steps and action items
- Any important considerations or warnings

**Format as a professional consulting summary.**

Conversation context:
{context}
"""
    
    return call_gemini(consensus_prompt, api_key_dict['gemini'])

# Main UI
st.title("🎭 Team of Rivals")
st.markdown("*Where AI consultants collaborate on your toughest challenges*")

st.markdown("Three AI consultants work together to understand your situation and develop solutions. They listen, learn, and build on each other's insights to give you perspectives no single AI could provide.")

# User participation guidance
with st.expander("💡 How to get the best results", expanded=False):
    st.markdown("""
    **You're an active participant, not a passive listener:**
    - **Speak freely** - ramble, think out loud, contradict yourself. That's how real insights emerge.
    - **Push back** if the discussion seems off track or missing something important.
    - **Ask follow-ups** when you want them to dig deeper or explore different angles.
    - **Use audio** for complex problems - it captures nuance and context that typing often misses.
    - **Don't worry about structure** - the consultants will help organize and focus the discussion.
    
    **Remember:** The best consultations happen when you stay engaged and guide the conversation toward what you actually need!
    """)

st.markdown("---")

# Consultation mode selection
st.subheader("🎯 Choose Your Consultation Style")
consultation_mode = st.radio(
    "How deep should we go?",
    ["🏃‍♂️ Quick tactical help (focused, efficient)", 
     "🔬 Deep strategic exploration (comprehensive, collaborative)"],
    key="mode_selection"
)

st.session_state.consultation_mode = "quick" if "Quick" in consultation_mode else "deep"

# Problem input with audio
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

# Text input
user_problem = st.text_area(
    "Or type your challenge here...",
    placeholder="Examples:\n- I have some Python code that feels messy\n- Need help planning a difficult team conversation\n- Trying to decide between two strategic directions\n- Working on a presentation that isn't clicking\n\nDon't worry about being perfectly clear - they'll ask questions!",
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

# Start consultation button
if st.button("🚀 Start AI Consultation", disabled=not combined_input or not api_available):
    
    # Initialize session
    st.session_state.session_start_time = time.time()
    st.session_state.last_checkin_time = None
    st.session_state.initial_responses_complete = False
    
    # Initialize conversation thread
    st.session_state.conversation_thread = [{
        "speaker": "User",
        "content": combined_input,
        "timestamp": "Initial request",
        "time": datetime.now()
    }]
    
    st.session_state.session_active = True
    st.rerun()

# Show active consultation interface
if st.session_state.session_active:
    
    # Check for Gemini moderation
    if check_for_gemini_moderation():
        st.markdown("---")
        st.subheader("🎯 5-Minute Pulse Check")
        
        if st.button("Get Gemini's Pulse Check"):
            api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
            
            with st.spinner("Gemini checking in..."):
                checkin_response = gemini_checkin(api_keys)
                
                st.session_state.conversation_thread.append({
                    "speaker": "Gemini (Moderator)",
                    "content": checkin_response,
                    "timestamp": "Pulse check",
                    "time": datetime.now()
                })
                
                st.session_state.last_checkin_time = time.time()
                
                st.markdown("### Gemini's Pulse Check:")
                st.markdown(checkin_response)
                st.rerun()
    
    # Display conversation
    st.markdown("---")
    st.subheader("💬 Consultation in Progress")
    
    # Show conversation thread
    for entry in st.session_state.conversation_thread:
        with st.container():
            if entry['speaker'] == 'User':
                st.markdown(f"**👤 You:** {entry['content']}")
            else:
                icon = "🤖" if "GPT-4" in entry['speaker'] else "🧠" if "Claude" in entry['speaker'] else "💎"
                st.markdown(f"**{icon} {entry['speaker']}:** {entry['content']}")
            st.markdown("---")
    
    # Only show "Continue the Conversation" after initial responses are complete
    if st.session_state.initial_responses_complete:
            # Persistent question input
            st.markdown("### 💬 Continue the Conversation")
            st.markdown("*Feel free to push back, ask follow-ups, or redirect the discussion!*")
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                follow_up = st.text_area(
                    "Ask a follow-up, share more context, or redirect:",
                    value=st.session_state.current_follow_up,
                    placeholder="Examples:\n- Can you see the actual code?\n- That's not quite right - here's what I meant...\n- I like that direction, but what about...\n- You're overcomplicating this - I just need...",
                    height=80,
                    key=f"followup_stable_{st.session_state.follow_up_counter}"
                )
            
            with col2:
                target_model = st.selectbox(
                    "Target:",
                    ["@everyone", "@gpt4", "@claude", "@gemini"],
                    key=f"target_stable_{st.session_state.follow_up_counter}"
                )
            
            if st.button("💬 Ask Question", key=f"ask_stable_{st.session_state.follow_up_counter}") and follow_up:
                
                # Add user question to thread
                st.session_state.conversation_thread.append({
                    "speaker": "User",
                    "content": f"Question for {target_model}: {follow_up}",
                    "timestamp": "Follow-up question",
                    "time": datetime.now()
                })
                
                # Determine which models to query
                if target_model == "@everyone":
                    models_to_query = ["GPT-4", "Claude", "Gemini"]
                    random.shuffle(models_to_query)  # Randomize order for each round
                elif target_model == "@gpt4":
                    models_to_query = ["GPT-4"]
                elif target_model == "@claude":
                    models_to_query = ["Claude"]
                elif target_model == "@gemini":
                    models_to_query = ["Gemini"]
                
                api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
                
                # Get responses
                for model in models_to_query:
                    with st.spinner(f"{model} responding..."):
                        response = generate_model_response(follow_up, model, api_keys, is_initial=False)
                        
                        st.session_state.conversation_thread.append({
                            "speaker": model,
                            "content": response,
                            "timestamp": "Follow-up response",
                            "time": datetime.now()
                        })
                
                # Clear the input and increment counter for next question
                st.session_state.current_follow_up = ""
                st.session_state.follow_up_counter += 1
                
                st.rerun()
    
    # Session management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Generate Consensus Document"):
            api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
            
            with st.spinner("Generating consensus document..."):
                consensus = generate_consensus_document(api_keys)
                
                st.session_state.conversation_thread.append({
                    "speaker": "Team of Rivals (Consensus)",
                    "content": consensus,
                    "timestamp": "Final consensus",
                    "time": datetime.now()
                })
                
                st.markdown("### 📋 Consultation Summary")
                st.markdown(consensus)
    
    with col2:
        if st.button("🔄 Start New Consultation"):
            # Reset session
            for key in ['conversation_thread', 'session_active', 'session_start_time', 'last_checkin_time', 'initial_responses_complete']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

# Sequential initial responses for new sessions
if st.session_state.session_active and len(st.session_state.conversation_thread) == 1:
    st.markdown("---")
    st.subheader("🎭 AI Consultants Responding")
    
    # Get the user's input
    user_input = st.session_state.conversation_thread[0]['content']
    
    # Add consultation mode context
    mode_context = f"User requested: {st.session_state.consultation_mode} consultation style"
    
    api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
    
    # Sequential responses instead of parallel
    models = ["GPT-4", "Claude", "Gemini"]
    random.shuffle(models)  # Randomize order for initial responses
    
    for i, model in enumerate(models):
        with st.spinner(f"{model} thinking..."):
            response = generate_model_response(
                user_input,  # Remove the mode_context since it's now handled inside the function
                model, 
                api_keys, 
                is_initial=True
            )
        
        # Add to conversation thread immediately
        st.session_state.conversation_thread.append({
            "speaker": model,
            "content": response,
            "timestamp": "Initial response",
            "time": datetime.now()
        })
        
        icon = "🤖" if model == "GPT-4" else "🧠" if model == "Claude" else "💎"
        st.markdown(f"### {icon} {model}")
        st.markdown(response)
        
        # Small delay between responses to reinforce sequential nature
        if i < len(models) - 1:
            time.sleep(1)
    
    # Mark initial responses as complete and trigger rerun
    st.session_state.initial_responses_complete = True
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Team of Rivals** - Where AI consultants collaborate to solve your toughest challenges 🎭✨")
