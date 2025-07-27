# tor1_main.py - Web Search Implementation
# Team of Rivals: Complete implementation with Challenge Assumptions + Web Search

import streamlit as st
import google.generativeai as genai
import openai
import anthropic
import re
import json
import time
import random
import requests
from urllib.parse import quote
from typing import Dict, List, Optional
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Team of Rivals", 
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    google_search_key = st.secrets["GOOGLE_SEARCH_API_KEY"]
    google_cse_id = st.secrets["GOOGLE_CSE_ID"]
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

Remember: You're not alone - your collaborators will step up and challenge you if needed.

IMPORTANT: Never respond as if you are another model. Each model speaks only for themselves in collaborative discussions.
"""

# Quick Mode Reviewer Guidelines with Search
REVIEWER_GUIDELINES_WITH_SEARCH = """
Your role is to help users assess how much they can rely on the primary response, enhanced with web search capabilities. Think of yourself as their research assistant with access to current information.

**Your process:**
1. Evaluate the primary response for factual claims that may need verification
2. Conduct web search when you encounter information that is:
   - Time-sensitive or potentially outdated
   - Specific factual claims (names, dates, statistics, locations)
   - Complex topics that could benefit from current context
3. Provide your reliability assessment

**Start your response with a reliability assessment emoji:**
• 🟢 if the information looks reliable and you don't see significant concerns
• 🟡 if it's useful but worth verifying some points or adding context
• 🔴 if there are significant issues or the user should approach with caution

**After web search (when conducted):**
- Update your reliability assessment based on search findings
- Clearly indicate when you've verified or corrected information using web search
- Use phrases like "According to recent data..." or "My research confirms/contradicts..."

**What to look for:**
- Claims that seem uncertain or hard to verify
- Missing context that could change conclusions  
- Assumptions that might not hold in all cases
- Areas where current information would be valuable
- Information that contradicts what you found through search

Your goal is to put the user in a better position to assess the response using both analysis and current information.

IMPORTANT: Never respond as if you are another model. You are providing your own independent review enhanced with web research.
"""

# Gemini Deep Mode Guidelines
GEMINI_DEEP_MODE_GUIDELINES = """
**Your Role: Primary Researcher and Foundation-Setter**

You are the first responder in Deep Dive mode, tasked with providing a research-informed foundation for the discussion.

**Your process:**
1. Analyze the user's query for factual elements that would benefit from current information
2. Conduct targeted web searches when the query involves:
   - Current events, recent developments, or time-sensitive information
   - Specific factual claims that need verification
   - Complex topics where multiple perspectives would be valuable
3. Present your findings as a foundation for deeper analysis by your colleagues

**How to present search findings:**
- Synthesize search results with your own analysis - don't just summarize
- Offer multiple perspectives when relevant, highlighting key facts
- Leave room for further analysis by clearly indicating areas that warrant deeper exploration
- Use clear indicators when information comes from web search: "Recent reports suggest..." "Current data indicates..."

**Your goal:** Provide an informed starting point that enhances rather than constrains the collaborative discussion.

**Collaboration focus:** Your colleagues will build upon your research-informed foundation with their own analysis and insights. Present information in a way that invites further exploration rather than closing off discussion.

IMPORTANT: Never respond as if you are another model. You are providing your own research-enhanced perspective to start the collaborative analysis.
"""

# Enhanced Deep Mode Guidelines for GPT-4 and Claude
DEEP_MODE_ENHANCED_GUIDELINES = """
**DEEP DIVE MODE - Collaborative Analysis with Research Foundation:**
• Build upon Gemini's research-informed foundation with your own analysis and insights
• Use search findings as a springboard for deeper exploration, not as constraints
• When your analysis differs from search results, explain your reasoning clearly
• Bring in relevant knowledge and perspectives not covered in the search results
• Focus on implications, connections, and strategic analysis beyond the immediate facts
• Briefly acknowledge when building directly on Gemini's research, but integrate naturally
• Apply critical thinking to all information, including search results

**Key principle:** Use the research foundation to enhance your analysis while maintaining your independent perspective and expertise.
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

# Web search functions
def perform_web_search(query: str, num_results: int = 5) -> dict:
    """
    Perform a Google Custom Search and return results
    """
    try:
        search_url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': google_search_key,
            'cx': google_cse_id,
            'q': query,
            'num': num_results
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        search_data = response.json()
        
        # Extract useful information
        results = []
        if 'items' in search_data:
            for item in search_data['items']:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'displayLink': item.get('displayLink', '')
                })
        
        return {
            'success': True,
            'results': results,
            'total_results': search_data.get('searchInformation', {}).get('totalResults', '0')
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Search failed: {str(e)[:100]}...",
            'results': []
        }

def format_search_results_for_prompt(search_data: dict) -> str:
    """
    Format search results for inclusion in model prompts
    """
    if not search_data['success'] or not search_data['results']:
        return "Web search was unavailable or returned no results."
    
    formatted_results = "Recent web search findings:\n\n"
    
    for i, result in enumerate(search_data['results'][:5], 1):
        formatted_results += f"{i}. {result['title']}\n"
        formatted_results += f"   Source: {result['displayLink']}\n"
        formatted_results += f"   Summary: {result['snippet']}\n"
        formatted_results += f"   Link: {result['link']}\n\n"
    
    return formatted_results

def extract_search_query(user_input: str, conversation_context: str) -> str:
    """
    Extract relevant search terms from user input and context
    """
    query = user_input.strip()
    
    # Remove common conversational elements
    query = query.replace("Can you", "").replace("Please", "").replace("I need", "")
    query = query.replace("help me", "").replace("tell me", "").strip()
    
    # Limit query length for API
    if len(query) > 100:
        query = query[:100]
    
    return query

# Audio transcription function
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

def call_gemini_with_search(prompt, api_key, search_query=None):
    """
    Call Gemini API with optional web search integration
    """
    try:
        # Perform web search if query provided
        search_results = ""
        if search_query:
            search_data = perform_web_search(search_query)
            search_results = format_search_results_for_prompt(search_data)
        
        # Combine search results with prompt
        enhanced_prompt = prompt
        if search_results:
            enhanced_prompt = f"{prompt}\n\n{search_results}\n\nPlease incorporate relevant information from the web search results above into your response when applicable."
        
        # Call Gemini with enhanced prompt
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(
            enhanced_prompt,
            generation_config={
                'temperature': 0.3,
                'max_output_tokens': 1200,
            }
        )
        return response.text
        
    except Exception as e:
        return f"[Gemini with search temporarily unavailable: {str(e)[:50]}...]"

def generate_model_response_with_search(question: str, model_name: str, api_key_dict: Dict, 
                                       is_initial: bool = False, role: str = "primary", 
                                       is_challenger: bool = False) -> str:
    """Generate a model's response with conversation context and optional web search for Gemini"""
    
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
    
    # Select appropriate guidelines based on role and model
    if is_challenger:
        base_guidelines = ELEVENTH_MAN_GUIDELINES
    elif role == "reviewer" and model_name.lower() == "gemini":
        base_guidelines = REVIEWER_GUIDELINES_WITH_SEARCH
    elif model_name.lower() == "gemini" and st.session_state.consultation_mode == "deep":
        base_guidelines = GEMINI_DEEP_MODE_GUIDELINES
    elif st.session_state.consultation_mode == "deep":
        base_guidelines = DEEP_MODE_ENHANCED_GUIDELINES
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

    # Handle Gemini with search
    if model_name.lower() == "gemini":
        # Extract search query for Gemini
        search_query = extract_search_query(question, context)
        return call_gemini_with_search(full_prompt, api_key_dict['gemini'], search_query)
    
    # Call other APIs normally
    elif model_name.lower() in ["gpt-4", "gpt-4o"]:
        return call_openai(full_prompt, api_key_dict['openai'])
    elif model_name.lower() == "claude":
        return call_anthropic(full_prompt, api_key_dict['anthropic'])
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
    # Main UI with new text and layout
st.title("🎭 Team of Rivals")
st.markdown("*Let ChatGPT, Claude and Gemini collaborate to answer your questions or dig into your toughest challenges*")

# Sidebar with comprehensive explanation (standard width)
with st.sidebar:
    st.markdown("### Team of Rivals: Are Three Heads Better Than One?")
    st.markdown("We think so. This app tests that theory by putting you together with the world's top AI tools — ChatGPT, Claude and Gemini. Tired of uncertain AI answers? Try a new approach.")
    
    st.markdown("**Choose Your Mode:**")
    st.markdown("Quick Mode: One model answers, another checks.")
    with st.expander("Tell me more"):
        st.markdown("Need answers promptly? Quick Mode is like having a smart researcher always at hand — and another one to make sure the first isn't making something up! And if you need to go deeper, Deep Dive is a click away.")
    
    st.markdown("Deep Dive: Three models collaborate. They dig into your issue, then brainstorm solutions. A digital roundtable, with egos checked at the door.")
    with st.expander("How it works"):
        st.markdown("Got a complex conundrum? In Deep Dive mode, the three AI models put their heads together. First they Listen & Learn, asking probing questions and sometimes challenging your assumptions. Then they switch to Build, Execute and Check — coming up with solutions and backstopping each other. It's an experiment in digital collaboration — we're still testing the waters, but the process itself can be illuminating.")
    
    st.markdown("**AI Teamwork Benefits:**")
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
        # Simple display - you can enhance this later
        st.success("Thanks! Feedback noted.")
        st.info(f"**Feedback:** {feedback}")
        # TODO: Later you could email this to yourself or save to a file

st.markdown("---")

# Consultation mode selection
st.subheader("🎯 Choose Your Consultation Style")
consultation_mode = st.radio(
    "How deep should we go?",
    ["🏃‍♂️ Quick & Simple — one model, one answer plus one review that includes web search. Can switch to Deep mode later.",
     "🔬 Deep Dive — work with all three models to dig into your challenge and collaborate on solutions"],
    key="mode_selection"
)

st.session_state.consultation_mode = "quick" if "Quick" in consultation_mode else "deep"

# Model selection for Quick Mode
if st.session_state.consultation_mode == "quick":
    st.subheader("📍 Quick Mode Setup")
    selected_model = st.radio(
        "Pick one model to answer your question:",
        ["GPT-4", "Claude", "Gemini"],
        horizontal=True,
        help="A second model will automatically review the response with web search for reliability."
    )
else:
    st.subheader("📍 Deep Dive Mode")
    st.info("All three consultants will collaborate on your question.")
    selected_model = None

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
    if st.session_state.consultation_mode == "quick":
        st.subheader("💬 Consultation in Progress")
        st.markdown("*Reviewer model will perform web search that may add several seconds*")
    else:
        st.subheader("💬 Consultation in Progress")
        st.markdown("*Research-enhanced collaboration in progress*")
    
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
    
    # Continue conversation section
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
            
            # Add Quick Mode option for flexibility
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("")  # Empty space for alignment
            with col2:
                if st.button("🏃‍♂️ Switch to Quick Mode", help="Faster single answer + review"):
                    st.session_state.follow_up_mode = "quick"
        
        st.markdown("---")
        
        # Follow-up input
        follow_up = st.text_area(
            "Ask a follow-up, share more context, or redirect:",
            placeholder="Examples:\n- Can you be more specific about...\n- That's not quite right - here's what I meant...\n- I like that direction, but what about...\n- You're overcomplicating this - I just need...",
            height=80,
            key=f"followup_{st.session_state.follow_up_counter}",
            value=""
        )
        
        # Challenge Assumptions checkbox
        challenge_active = False
        if st.session_state.consultation_mode == "deep":
            challenge_active = st.checkbox(
                "Activate **Challenge Assumptions** mode for wider viewpoints?",
                key=f"challenge_assumptions_{st.session_state.follow_up_counter}",
                help="Uses the 'Eleventh Man' approach - one model questions consensus to prevent groupthink"
            )
        
        # Pro tip for steering
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
                # Quick Mode: use assigned responder and Gemini reviewer
                if not st.session_state.quick_responder:
                    responder, _ = assign_quick_mode_roles(selected_model or "GPT-4")
                    st.session_state.quick_responder = responder
                    st.session_state.quick_reviewer = "Gemini"
                else:
                    responder = st.session_state.quick_responder
                
                # Get primary response
                with st.spinner(f"{responder} responding..."):
                    response = generate_model_response_with_search(follow_up, responder, api_keys, is_initial=False)
                    st.session_state.conversation_thread.append({
                        "speaker": responder,
                        "content": response,
                        "timestamp": "Follow-up response",
                        "time": datetime.now()
                    })
                
                # Get Gemini reviewer response with search
                with st.spinner("💎 Gemini reviewing with web search..."):
                    review_prompt = f"Review this response for reliability: {response}"
                    review_response = generate_model_response_with_search(review_prompt, "Gemini", api_keys, is_initial=False, role="reviewer")
                    st.session_state.conversation_thread.append({
                        "speaker": "Gemini [Reviewer]",
                        "content": review_response,
                        "timestamp": "Follow-up review",
                        "time": datetime.now()
                    })
            
            else:
                # Deep Dive Mode with optional Challenge Assumptions
                models = ["Gemini", "GPT-4", "Claude"]  # Gemini first
                
                if challenge_active:
                    # Challenge Assumptions mode - one model becomes challenger
                    challenger_model = random.choice(["GPT-4", "Claude"])  # Not Gemini since it goes first
                    other_models = [m for m in models if m != challenger_model]
                    
                    # Standard collaborators respond
                    for model in other_models:
                        if model == "Gemini":
                            with st.spinner("💎 Gemini researching and analyzing..."):
                                response = generate_model_response_with_search(follow_up, model, api_keys, is_initial=False)
                        else:
                            with st.spinner(f"{model} building on the research..."):
                                response = generate_model_response_with_search(follow_up, model, api_keys, is_initial=False)
                        
                        st.session_state.conversation_thread.append({
                            "speaker": model,
                            "content": response,
                            "timestamp": "Follow-up response",
                            "time": datetime.now()
                        })
                    
                    # Challenger responds last
                    with st.spinner(f"{challenger_model} challenging assumptions..."):
                        response = generate_model_response_with_search(follow_up, challenger_model, api_keys, is_initial=False, is_challenger=True)
                        st.session_state.conversation_thread.append({
                            "speaker": f"{challenger_model} [Challenge Assumptions]",
                            "content": response,
                            "timestamp": "Challenger response",
                            "time": datetime.now()
                        })
                
                else:
                    # Standard Deep Dive - Gemini first, then others
                    for model in models:
                        if model == "Gemini":
                            with st.spinner("💎 Gemini researching and analyzing..."):
                                response = generate_model_response_with_search(follow_up, model, api_keys, is_initial=False)
                        else:
                            with st.spinner(f"{model} building on the research..."):
                                response = generate_model_response_with_search(follow_up, model, api_keys, is_initial=False)
                        
                        st.session_state.conversation_thread.append({
                            "speaker": model,
                            "content": response,
                            "timestamp": "Follow-up response",
                            "time": datetime.now()
                        })
            
            # Clear input and reset state
            st.session_state.follow_up_counter += 1
            st.session_state.follow_up_mode = None
            st.rerun()
    
    # Session management
    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        # Reset session
        for key in ['conversation_thread', 'session_active', 'round_number', 'follow_up_counter', 'current_follow_up', 'quick_responder', 'quick_reviewer', 'follow_up_mode', 'audio_transcription', 'input_text']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Initial responses
if st.session_state.session_active and len(st.session_state.conversation_thread) == 1:
    st.markdown("---")
    st.subheader("🎭 AI Consultants Responding")
    
    user_input = st.session_state.conversation_thread[0]['content']
    api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
    
    if st.session_state.consultation_mode == "quick":
        # Quick Mode: Single responder + Gemini reviewer with search
        responder, _ = assign_quick_mode_roles(selected_model)
        st.session_state.quick_responder = responder
        st.session_state.quick_reviewer = "Gemini"
        
        # Primary response
        with st.spinner(f"{responder} thinking..."):
            response = generate_model_response_with_search(user_input, responder, api_keys, is_initial=True)
        
        st.session_state.conversation_thread.append({
            "speaker": responder,
            "content": response,
            "timestamp": "Initial response",
            "time": datetime.now()
        })
        
        icon = "🤖" if responder == "GPT-4" else "🧠" if responder == "Claude" else "💎"
        st.markdown(f"### {icon} {responder}")
        st.markdown(response)
        
        # Gemini reviewer response with search
        with st.spinner("💎 Gemini reviewing with web search..."):
            review_prompt = f"Review this response for reliability: {response}"
            review_response = generate_model_response_with_search(review_prompt, "Gemini", api_keys, is_initial=True, role="reviewer")
        
        st.session_state.conversation_thread.append({
            "speaker": "Gemini [Reviewer]",
            "content": review_response,
            "timestamp": "Initial review",
            "time": datetime.now()
        })
        
        # Display review with proper emoji parsing
        emoji, content, explanation = parse_reviewer_response(review_response)
        st.markdown(f"### 🔍 Review by 💎 Gemini")
        st.markdown(f"{emoji} {content}")
    
    else:
        # Deep Dive Mode: Gemini first with search, then others
        models = ["Gemini", "GPT-4", "Claude"]
        
        for i, model in enumerate(models):
            if model == "Gemini":
                with st.spinner("💎 Gemini researching and analyzing..."):
                    response = generate_model_response_with_search(user_input, model, api_keys, is_initial=True)
                st.markdown("*Research-enhanced analysis:*")
            else:
                with st.spinner(f"{model} building on the research..."):
                    response = generate_model_response_with_search(user_input, model, api_keys, is_initial=True)
            
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
