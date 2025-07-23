# tor1_main.py - Hybrid Architecture: v15 Foundation + Key Innovations
# Combines working v15 direct execution with Quick Mode reviewer system and steering controls

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

# Initialize session state (simplified from RC10, based on v15)
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
# Steering controls
if 'steering_command' not in st.session_state:
    st.session_state.steering_command = ""
if 'steering_command_used' not in st.session_state:
    st.session_state.steering_command_used = False

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

# Quick Mode Reviewer Guidelines (from modular version)
REVIEWER_GUIDELINES = """
Your role is to help users assess how much they can rely on the primary response. Think of yourself as their research assistant, spotting elements that deserve caution or further verification.

**Consider starting your response with a reliability assessment:**
• 🟢 if the information looks reliable and you don't see significant concerns
• 🟡 if it's useful but worth verifying some points or adding context
• 🔴 if there are significant issues or the user should approach with caution

**Examples of helpful reliability assessments:**

🟢 "This covers the key points well - the theories mentioned are the commonly accepted explanations, and the verification suggestions are appropriate."

🟡 "Good framework, but this doesn't address the transition process for current visiting students - you'll want to contact admissions about simplified procedures for your specific situation."

🔴 "Several specific claims here need verification. The 2025 salary projections don't specify data sources, and the market predictions assume stable economic conditions. Check current industry reports before relying on these numbers."

**What to look for:**
- Claims that seem uncertain or hard to verify
- Missing context that could change conclusions  
- Assumptions that might not hold in all cases
- Areas where the user should do additional research
- Information that contradicts what you know (share your perspective for comparison)

Your goal is to put the user in a better position to assess the response. Focus on being helpful - whether that's flagging concerns or confirming the response looks solid.

IMPORTANT: Never respond as if you are another model. You are providing your own independent review.
"""

# Steering Controls Functions
def show_steering_controls():
    """Display conversation steering buttons."""
    st.markdown("### 🎯 Steer the Conversation")
    st.caption("Quick ways to redirect or refocus the discussion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔍 Clarity & Focus**")
        if st.button("Be more direct", help="Ask for clearer, more direct responses"):
            st.session_state.steering_command = "Please be more direct and cut to the point."
        if st.button("I don't understand", help="Ask for clearer explanation"):
            st.session_state.steering_command = "I don't understand. Could you rephrase that more clearly?"
        if st.button("Focus on core issue", help="Get back to the main problem"):
            st.session_state.steering_command = "Please focus on the core issue, not the surrounding details."
    
    with col2:
        st.markdown("**🎯 Depth & Direction**")
        if st.button("Go deeper", help="Explore the last point in more detail"):
            st.session_state.steering_command = "Please go deeper on that last point."
        if st.button("Focus on solutions", help="Move toward actionable recommendations"):
            st.session_state.steering_command = "Let's focus on finding a solution now."
        if st.button("Give me recommendations", help="Ask for clear next steps"):
            st.session_state.steering_command = "Please give me clear recommendations."
    
    # Tone check section
    st.markdown("**🙅‍♂️ Tone Check**")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("Less flattery", help="Ask for honest assessment instead of praise"):
            st.session_state.steering_command = "Please be honest and cut down on praise or flattery."
    with col4:
        if st.button("More disagreement", help="Encourage alternative viewpoints"):
            st.session_state.steering_command = "I'd like to hear more disagreement or alternative takes."
    
    # Show current steering command if one was selected
    if st.session_state.get("steering_command"):
        st.info(f"🎯 **Steering signal:** \"{st.session_state.steering_command}\"")
        st.caption("This will be included in the next model prompt automatically.")
        return True
    
    return False

def get_steering_command():
    """Get and mark steering command as used."""
    command = st.session_state.get("steering_command", "")
    if command:
        st.session_state.steering_command_used = True
    return command

def clear_steering_command():
    """Clear steering command after use."""
    if "steering_command" in st.session_state:
        del st.session_state.steering_command
    st.session_state.steering_command_used = False

# API calling functions (from v15, enhanced for Claude issues)
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
        
        # Enhanced handling for Claude's sensitivity issues
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

def generate_model_response(question: str, model_name: str, api_key_dict: Dict, is_initial: bool = False, role: str = "primary") -> str:
    """Generate a model's response with conversation context and guidelines"""
    
    # Get recent context (from v15 approach)
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
    if role == "reviewer":
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
    
    # Add steering command if present
    steering = get_steering_command()
    if steering:
        steering_guidance = f"\n\n**User Steering Request:** {steering}"
        clear_steering_command()
    else:
        steering_guidance = ""
    
    # Create full prompt
    full_prompt = f"""{base_guidelines}

{mode_guidance}

**Recent Conversation Context:**
{context}

**Current Question/Topic:** {question}{steering_guidance}

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
    lines = response.split('\n', 1)
    first_line = lines[0].strip()
    explanation = lines[1].strip() if len(lines) > 1 else ""
    
    # Check for reliability emojis
    reliability_emojis = {
        "🟢": "No significant concerns",
        "🟡": "Worth verifying some points", 
        "🔴": "Significant issues - approach with caution"
    }
    
    for emoji, description in reliability_emojis.items():
        if first_line.startswith(emoji):
            remaining = first_line[len(emoji):].strip()
            if remaining:
                return emoji, remaining, explanation
            else:
                return emoji, description, explanation
    
    # Fallback
    return "🔍", first_line, explanation

# Main UI (adapted from v15 with Quick/Deep mode selection)
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
    - **Use steering controls** to redirect the conversation when needed.
    - **Don't worry about structure** - the consultants will help organize and focus the discussion.
    
    **Remember:** The best consultations happen when you stay engaged and guide the conversation toward what you actually need!
    """)

st.markdown("---")

# Consultation mode selection (Quick vs Deep)
st.subheader("🎯 Choose Your Consultation Style")
consultation_mode = st.radio(
    "How deep should we go?",
    ["🏃‍♂️ Quick & Simple — with backup. Fast, direct answers with a second model reviewing for accuracy.",
     "🔬 Deep strategic exploration (comprehensive, collaborative)"],
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
        help="A second model will automatically review the response for reliability."
    )
else:
    st.subheader("📍 Deep Dive Mode")
    st.info("All three consultants will collaborate on your question.")
    selected_model = None

# Problem input
st.subheader("📢 Share Your Challenge")
st.markdown("**Think out loud:** Complex problems often need rambling to understand properly. Don't worry about being perfectly clear - the consultants will ask follow-up questions!")

# Text input
user_problem = st.text_area(
    "Describe your challenge:",
    placeholder="Examples:\n- I have some Python code that feels messy\n- Need help planning a difficult team conversation\n- Trying to decide between two strategic directions\n- Working on a presentation that isn't clicking\n\nDon't worry about being perfectly clear - they'll ask questions!",
    height=120,
    key="problem_input"
)

# Start consultation button (v15 direct execution style)
if st.button("🚀 Start AI Consultation", disabled=not user_problem or not api_available):
    
    # Initialize conversation thread (v15 style)
    st.session_state.conversation_thread = [{
        "speaker": "User",
        "content": user_problem,
        "timestamp": "Initial request",
        "time": datetime.now()
    }]
    
    st.session_state.session_active = True
    st.session_state.round_number = 1
    st.rerun()

# Show active consultation interface
if st.session_state.session_active:
    
    # Display conversation (v15 style)
    st.markdown("---")
    st.subheader("💬 Consultation in Progress")
    
    # Show conversation thread
    for entry in st.session_state.conversation_thread:
        with st.container():
            if entry['speaker'] == 'User':
                st.markdown(f"**👤 You:** {entry['content']}")
            elif entry['speaker'].endswith(' [Reviewer]'):
                # Show reviewer responses with special formatting
                reviewer_name = entry['speaker'].replace(' [Reviewer]', '')
                icon = "🤖" if "GPT-4" in reviewer_name else "🧠" if "Claude" in reviewer_name else "💎"
                st.markdown(f"**🔍 Review by {icon} {reviewer_name}:** {entry['content']}")
            else:
                icon = "🤖" if "GPT-4" in entry['speaker'] else "🧠" if "Claude" in entry['speaker'] else "💎"
                st.markdown(f"**{icon} {entry['speaker']}:** {entry['content']}")
            st.markdown("---")
    
    # Continue conversation section with steering controls
    if len(st.session_state.conversation_thread) > 1:  # After initial responses
        st.markdown("### 💬 Continue the Conversation")
        st.markdown("*Feel free to push back, ask follow-ups, or redirect the discussion!*")
        
        # Show steering controls
        show_steering_controls()
        st.markdown("---")
        
        # Mode switching options
        current_mode = st.session_state.consultation_mode
        
        col1, col2 = st.columns(2)
        with col1:
            if current_mode == "quick":
                mode_button_text = "🏃‍♂️ Quick Follow-up"
                mode_help = "Same model + reviewer approach"
            else:
                mode_button_text = "🔬 Deep Dive Follow-up"  
                mode_help = "All consultants collaborate"
        
        with col2:
            if current_mode == "quick":
                if st.button("🔬 Switch to Deep Dive", help="Get all three models collaborating"):
                    st.session_state.consultation_mode = "deep"
                    st.rerun()
            else:
                if st.button("🏃‍♂️ Quick Question", help="Single model + reviewer"):
                    st.session_state.consultation_mode = "quick"
                    st.rerun()
        
        # Follow-up input (v15 direct execution style)
        follow_up = st.text_area(
            "Ask a follow-up, share more context, or redirect:",
            value=st.session_state.current_follow_up,
            placeholder="Examples:\n- Can you be more specific about...\n- That's not quite right - here's what I meant...\n- I like that direction, but what about...\n- You're overcomplicating this - I just need...",
            height=80,
            key=f"followup_{st.session_state.follow_up_counter}"
        )
        
        if st.button("💬 Ask Follow-up", key=f"ask_{st.session_state.follow_up_counter}") and follow_up:
            
            # Add user question to thread
            st.session_state.conversation_thread.append({
                "speaker": "User",
                "content": follow_up,
                "timestamp": "Follow-up question",
                "time": datetime.now()
            })
            
            api_keys = {'openai': openai_key, 'anthropic': anthropic_key, 'gemini': gemini_key}
            
            # Determine models to query based on mode
            if st.session_state.consultation_mode == "quick":
                # Quick Mode: use assigned responder and reviewer
                if not st.session_state.quick_responder:
                    # First quick follow-up, need to assign
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
                # Deep Dive Mode: all models respond
                models = ["GPT-4", "Claude", "Gemini"]
                random.shuffle(models)
                
                for model in models:
                    with st.spinner(f"{model} responding..."):
                        response = generate_model_response(follow_up, model, api_keys, is_initial=False)
                        st.session_state.conversation_thread.append({
                            "speaker": model,
                            "content": response,
                            "timestamp": "Follow-up response",
                            "time": datetime.now()
                        })
            
            # Clear input and increment counter
            st.session_state.current_follow_up = ""
            st.session_state.follow_up_counter += 1
            st.rerun()
    
    # Session management
    st.markdown("---")
    if st.button("🔄 Start New Consultation"):
        # Reset session (v15 style)
        for key in ['conversation_thread', 'session_active', 'round_number', 'follow_up_counter', 'current_follow_up', 'quick_responder', 'quick_reviewer']:
            if key in st.session_state:
                del st.session_state[key]
        clear_steering_command()
        st.rerun()

# Initial responses (v15 direct execution style)
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
        
        reviewer_icon = "🤖" if reviewer == "GPT-4" else "🧠" if reviewer == "Claude" else "💎"
        st.markdown(f"### 🔍 Review by {reviewer_icon} {reviewer}")
        st.markdown(review_response)
    
    else:
        # Deep Dive Mode: All models respond (v15 style)
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
            
            # Small delay between responses (v15 style)
            if i < len(models) - 1:
                time.sleep(1)
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown("**Team of Rivals** - Where AI consultants collaborate to solve your toughest challenges 🎭✨")
