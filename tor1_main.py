# tor1_main.py - v1.6 - Final UI Header Fix
# This version corrects the disclaimer header size to match other headers.

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import random
import time
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
except Exception as e:
    st.error(f"API keys not configured: {e}")
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

# ----------------------
# Sidebar
# ----------------------
with st.sidebar:
    st.markdown("### Team of Rivals: Are Three Heads Better Than One?")
    st.markdown("We think so. This app tests that theory by putting you together with the world's top AI tools — ChatGPT, Claude and Gemini. Tired of uncertain AI answers? Try a new approach.")
    
    st.markdown("AI Teamwork Benefits:")
    st.markdown("Error Reduction: Multiple AIs catch more mistakes or hallucinations. Not perfect, but better.")
    with st.expander("Learn more"):
        st.markdown("Our AI models work together to catch errors and inconsistencies, much like a team of dedicated proofreaders. While no system is perfect, this collaborative approach significantly reduces the likelihood of inaccuracies.")
    
    st.markdown("Idea Synthesis: Diverse perspectives lead to nuanced solutions.")
    with st.expander("See it in action"):
        st.markdown("Imagine a roundtable of AI consultants brainstorming solutions to your problem. By combining their diverse perspectives and expertise, they can often generate more creative and nuanced solutions than any single AI could achieve alone. An optional Challenge Assumptions mode can make sure ideas are tested in a constructive way.")
    
    st.markdown("You're in Charge: Capable AI team, but you call the shots.")
    with st.expander("Learn to direct"):
        st.markdown("You're not just a passive observer; you're running this AI-powered think tank. Your feedback, questions, and insights guide the models towards the most relevant and helpful outcomes. You're in control of the process.")
    
    st.markdown("**Getting the Most Out of This Tool:**")
    st.markdown("Be Sharp - or Fuzzy: Specific instructions can help hone AI models' output, but for getting started sometimes exploratory rambling is more effective. Our audio input lets you talk through a problem, like free-associating with a highly attentive colleague.")
    with st.expander("Effective prompting tips"):
        st.markdown("While specific prompts often work best, our audio input recognizes that sometimes exploratory rambling uncovers complex issues. Like a newspaper editor encouraging a reporter to free-associate, talking through a problem can surface hidden insights. Use clear prompts when possible, but don't hesitate to think out loud for thornier issues.")
    
    st.markdown("You're in Charge: Think of AI as a brilliant intern—lots of potential, but still learning. Your critical thinking is essential.")
    with st.expander("Understanding AI limitations"):
        st.markdown("AI is a rapidly evolving technology, but it's not magic. Our models are constantly learning, but they can still make mistakes or exhibit biases. Your critical thinking skills are essential for evaluating the information provided and ensuring responsible use of AI.")
    
    st.markdown("Join the Experiment: We're testing multi-AI collaboration. Your feedback helps shape its evolution.")
    with st.expander("Feedback & Beta Program"):
        st.markdown("We're in the early stages of exploring collaborative AI potential. Your feedback shapes this technology's future. Join our beta program to be part of this journey, but remember: this is an experiment. Results may vary, and we're learning alongside you.")

    st.markdown("---")
    st.markdown("**Beta Feedback**")
    
    feedback = st.text_area(
        "How's it working? What would you change?",
        placeholder="Share your thoughts: What worked well? What was confusing? What features would you want?",
        height=80,
        key="beta_feedback"
    )
    
    if st.button("Send Feedback", disabled=not feedback.strip()):
        st.success("Thanks! Feedback noted.")
# ----------------------
# Main UI
# ----------------------
st.markdown("<h1 style='margin-bottom: 0;'>Team of Rivals</h1>", unsafe_allow_html=True)
st.markdown("*ChatGPT, Claude, Gemini. Three top AI minds. One collaborative challenge.*")

st.markdown("### 👋 All-Too-Candid Opening Disclaimer")
st.markdown("""
Is this tool worth using? That depends! If your question is simple, you may well get faster, stronger replies using the free tools at [chatgpt.com](https://chatgpt.com), [claude.ai](https://claude.ai), or [gemini.google.com](https://gemini.google.com). (Though here's a tip we learned along the way -- to be more certain your answer isn't made up, cut and paste the AI's output into another AI and ask, "Does this seem right?")
""")
st.markdown("### 🤔 So What’s This Tool For?")
st.markdown("""
**Complicated, layered questions.** The kind that benefit from disagreement, reflection, and synthesis. 
Here, the models aren’t just answering — they’re working together. Sometimes it adds up to more than the sum of the parts.

So is this app truly better for complex, layered questions than just using one model on its own?  

**We think so — but help us find out.**
""")

st.markdown("---")
st.subheader("📢 Share Your Challenge -- talk it out or type it in")
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
                icon = "🤖" if "GPT-4" in challenger_name else "🧠" if "Claude" in challenger_name else "💎"
                st.markdown(f"**🕵️ {icon} {challenger_name} (Challenge Assumptions):** {entry['content']}")
            else:
                icon = "🤖" if "GPT-4" in entry['speaker'] else "🧠" if "Claude" in entry['speaker'] else "💎"
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
            models = ["GPT-4", "Claude", "Gemini"]
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
    models = ["GPT-4", "Claude", "Gemini"]
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
st.markdown("<p style='text-align: center; font-size: 0.85em; color: gray;'>Team of Rivals v1.6 — UI Polish</p>", unsafe_allow_html=True)
