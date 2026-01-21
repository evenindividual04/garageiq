"""
GarageIQ — Premium Demo UI
Enhanced UI/UX with sample inputs, animated results, and polished design.
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# CRITICAL: Load .env BEFORE importing anything from automotive_intent
# Otherwise config.py will read empty env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Force config to reload from environment
import os
os.environ.setdefault("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

import streamlit as st
from automotive_intent.core.schemas import ClassificationRequest
from automotive_intent.pipeline import create_pipeline
from automotive_intent.services.entities import get_entity_extractor
from automotive_intent.services.transcriber import get_transcriber
from automotive_intent.services.reporting import get_report_generator
from audio_recorder_streamlit import audio_recorder

# Page config
st.set_page_config(
    page_title="GarageIQ",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

# Premium CSS with glassmorphism and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .stApp { 
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #0f1729 100%); 
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1rem;
    }
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .subtitle { 
        color: #94a3b8; 
        font-size: 1.1rem; 
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .subtitle-highlight { color: #a78bfa; font-weight: 600; }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Status badges */
    .status-confirmed { 
        background: linear-gradient(135deg, #10B981 0%, #059669 100%); 
        color: white; padding: 0.5rem 1.2rem; border-radius: 50px; 
        font-weight: 700; font-size: 0.9rem; display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
    }
    .status-ambiguous { 
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%); 
        color: white; padding: 0.5rem 1.2rem; border-radius: 50px; font-weight: 700;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    .status-out-of-scope { 
        background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%); 
        color: white; padding: 0.5rem 1.2rem; border-radius: 50px; font-weight: 700;
    }
    
    /* Intent path */
    .intent-path { 
        font-family: 'JetBrains Mono', 'Fira Code', monospace; 
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        padding: 1rem 1.5rem; 
        border-radius: 12px; 
        color: #c4b5fd; 
        font-size: 1rem; 
        border-left: 4px solid #8b5cf6;
        margin: 1rem 0;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(139, 92, 246, 0.1);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #a78bfa; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }
    
    /* Language badge */
    .lang-badge { 
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
        color: #a5b4fc; 
        padding: 0.4rem 1rem; 
        border-radius: 20px; 
        font-size: 0.8rem; 
        font-weight: 600;
        border: 1px solid rgba(139, 92, 246, 0.3);
        display: inline-block;
    }
    
    /* Sample buttons */
    .sample-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        color: #94a3b8;
        cursor: pointer;
        transition: all 0.2s;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .sample-btn:hover {
        background: rgba(139, 92, 246, 0.15);
        border-color: rgba(139, 92, 246, 0.3);
        color: #c4b5fd;
    }
    
    /* Inputs */
    .stTextArea textarea { 
        background: rgba(255, 255, 255, 0.05) !important; 
        border: 1px solid rgba(255, 255, 255, 0.1) !important; 
        border-radius: 12px !important; 
        color: white !important;
        font-size: 1rem !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(139, 92, 246, 0.5) !important;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1) !important;
    }
    
    /* Buttons */
    .stButton > button { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; 
        color: white !important; 
        border: none !important; 
        border-radius: 50px !important; 
        padding: 0.7rem 2rem !important; 
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1729 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #93c5fd;
        font-size: 0.9rem;
    }
    
    /* Similar tickets */
    .ticket-item {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    
    /* Confidence bar */
    .confidence-bar {
        background: rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
</style>
""", unsafe_allow_html=True)


# Sample inputs for quick testing
SAMPLE_INPUTS = [
    {"text": "Brake lagane par awaaz aa rahi hai", "label": "Brake Noise (Hindi)"},
    {"text": "Scooty on nahi ho rahi", "label": "Scooty Won't Start"},
    {"text": "AC thanda nahi kar raha", "label": "AC Not Cooling"},
    {"text": "CNG par switch nahi ho raha", "label": "CNG Issue"},
    {"text": "Engine overheating on highway", "label": "Overheating"},
    {"text": "Getting P0420 code", "label": "DTC Code"},
]


def detect_language(text: str) -> tuple:
    """Detect language with confidence."""
    try:
        from langdetect import detect, detect_langs
        probs = detect_langs(text)
        if probs:
            top = probs[0]
            return top.lang, top.prob
    except:
        pass
    
    # Fallback: check for Hindi characters
    hindi_chars = set("अआइईउऊएऐओऔकखगघचछजझटठडढणतथदधनपफबभमयरलवशषसह")
    if any(c in hindi_chars for c in text):
        return "hi", 0.9
    
    # Check for Hinglish patterns
    hinglish_words = ["gaadi", "gadi", "nahi", "hai", "mein", "kya", "kar", "raha", "rahi", "ho", "lagane", "awaaz"]
    if any(w in text.lower() for w in hinglish_words):
        return "hi", 0.7
    
    return "en", 0.8


def init_pipeline():
    """Initialize pipeline once."""
    if st.session_state.pipeline is None:
        try:
            st.session_state.pipeline = create_pipeline(use_ollama=True, use_nllb=False)
        except Exception as e:
            st.error(f"Pipeline error: {e}")


def add_to_history(text: str, result: dict, response_time: float, language: str):
    """Add classification to history."""
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "text": text[:50] + "..." if len(text) > 50 else text,
        "status": result.get("classification_status", "UNKNOWN"),
        "language": language,
        "response_time_ms": int(response_time * 1000),
        "mode": result.get("mode", "fast"),
        "result": result
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:15]


def main():
    init_pipeline()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # Mode selector with better styling
        mode = st.radio(
            "Analysis Mode",
            ["⚡ Fast Pipeline", "🧠 Multi-Agent (RAG)"],
            index=0,
            help="**Fast:** Single LLM call, ~3-5s\n\n**Multi-Agent:** 4 agents with RAG, ~10-15s"
        )
        st.session_state["mode"] = "fast" if "Fast" in mode else "advanced"
        
        st.markdown("---")
        
        # History section
        st.markdown("## 📜 History")
        
        if st.session_state.history:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", len(st.session_state.history))
            with col2:
                confirmed = sum(1 for h in st.session_state.history if h["status"] == "CONFIRMED")
                st.metric("Confirmed", confirmed)
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        
        st.markdown("")
        
        for entry in st.session_state.history[:6]:
            status_emoji = {"CONFIRMED": "●", "AMBIGUOUS": "○", "OUT_OF_SCOPE": "—"}.get(entry["status"], "?")
            mode_icon = "⚡" if entry.get("mode") == "fast" else "🧠"
            with st.expander(f"{status_emoji} {entry['text'][:30]}...", expanded=False):
                st.caption(f"{mode_icon} {entry['response_time_ms']}ms • 🌐 {entry['language'].upper()}")
        
        st.markdown("---")
        st.markdown("##### Enterprise Edition 🚀")
        st.caption("Powered by Groq + LangGraph")
    
    # Main content
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">GarageIQ</h1>
        <p class="subtitle">Multilingual Service Complaint Classification • <span class="subtitle-highlight">Hindi • English • Hinglish</span></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two column layout
    col_input, col_result = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("### Enter Complaint")
        
        # Text input
        # Text input with Voice option
        
        # Initialize session state
        if "transcribed_input" not in st.session_state:
            st.session_state.transcribed_input = ""
        if "sample_input" not in st.session_state:
            st.session_state.sample_input = ""
        
        # Check if sample was clicked (before rendering text area)
        default_text = st.session_state.sample_input or st.session_state.transcribed_input
        
        # Voice Input
        col_mic, col_txt = st.columns([1, 6])
        with col_mic:
            st.markdown("<br>", unsafe_allow_html=True)
            audio_bytes = audio_recorder(
                text="",
                recording_color="#ef4444",
                neutral_color="#6b7280",
                icon_size="2rem",
                key="voice_recorder"
            )
        
        # Transcribe if audio recorded
        if audio_bytes and audio_bytes != st.session_state.get("last_audio"):
            st.session_state.last_audio = audio_bytes  # Prevent re-processing same audio
            try:
                transcriber = get_transcriber()
                with st.spinner("Transcribing..."):
                    transcribed_text = transcriber.transcribe(audio_bytes)
                    if transcribed_text:
                        st.session_state.transcribed_input = transcribed_text
                        st.session_state.sample_input = ""  # Clear any sample
                        st.rerun()  # Rerun to populate text area
                    else:
                        st.warning("⚠️ Could not transcribe audio. Check your GROQ_API_KEY.")
            except Exception as e:
                st.error(f"❌ Transcription error: {str(e)}")

        # Get default value from session state
        default_text = st.session_state.sample_input or st.session_state.transcribed_input
        
        with col_txt:
            input_text = st.text_area(
                label="Complaint",
                value=default_text,
                placeholder="Describe your vehicle problem... (or tap mic to speak)\n\nExamples:\n• My car won't start\n• Gadi garam ho rahi hai\n• Scooty on nahi ho rahi",
                height=140,
                label_visibility="collapsed"
            )
        
        # Update session state if user manually edits
        if input_text and input_text != default_text:
            st.session_state.transcribed_input = ""
            st.session_state.sample_input = ""
        
        # Language detection badge
        if input_text.strip():
            lang, conf = detect_language(input_text)
            lang_name = {"hi": "🇮🇳 Hindi", "en": "🇬🇧 English", "mr": "🇮🇳 Marathi"}.get(lang, lang.upper())
            st.markdown(f'<span class="lang-badge">{lang_name} • {int(conf*100)}% confidence</span>', unsafe_allow_html=True)
        
        # Analyze button
        st.markdown("")
        col_btn, _ = st.columns([1, 2])
        with col_btn:
            classify_btn = st.button("Analyze", use_container_width=True)
        
        # Sample inputs
        st.markdown("---")
        st.markdown("##### Sample Inputs")
        
        sample_cols = st.columns(3)
        for i, sample in enumerate(SAMPLE_INPUTS):
            with sample_cols[i % 3]:
                if st.button(sample["label"], key=f"sample_{i}", use_container_width=True):
                    st.session_state.sample_input = sample["text"]
                    st.session_state.transcribed_input = ""  # Clear transcription
                    st.rerun()
        
        # Clear sample input after it's been used (to avoid persistence)
        if st.session_state.sample_input and input_text == st.session_state.sample_input:
            pass  # Keep it while displayed
        elif st.session_state.sample_input:
            st.session_state.sample_input = ""  # Clear after use
    
    with col_result:
        st.markdown("### Analysis Result")
        
        if classify_btn and input_text.strip():
            if st.session_state.pipeline is None:
                st.error("Pipeline not ready. Please wait...")
                return
            
            lang, lang_conf = detect_language(input_text)
            use_advanced = st.session_state.get("mode") == "advanced"
            
            # Fun loading messages to make wait feel shorter
            loading_tips = [
                "Tip: Regular oil changes extend engine life by 30%",
                "Tip: Check tire pressure monthly for better mileage",
                "Analyzing vehicle systems...",
                "Processing diagnostic data...",
                "Searching knowledge base...",
                "Matching with historical cases...",
                "Generating recommendations...",
                "Finalizing diagnosis...",
            ]
            
            # Engaging progress display
            start_time = time.time()
            progress_placeholder = st.empty()
            tip_placeholder = st.empty()
            status_placeholder = st.empty()
            
            import random
            current_tip = random.choice(loading_tips)
            last_tip_change = time.time()
            
            def update_progress(step, message, progress_pct):
                nonlocal current_tip, last_tip_change
                
                elapsed = time.time() - start_time
                
                # Change tip every 8 seconds
                if elapsed - last_tip_change > 8:
                    current_tip = random.choice(loading_tips)
                    last_tip_change = elapsed
                
                with progress_placeholder:
                    st.progress(progress_pct, text=f"Step {step}: {message}")
                with tip_placeholder:
                    st.info(current_tip)
                with status_placeholder:
                    # Show encouraging message based on time
                    if elapsed < 10:
                        msg = f"⏱️ {elapsed:.0f}s • Analyzing your complaint..."
                    elif elapsed < 30:
                        msg = f"⏱️ {elapsed:.0f}s • AI is thinking deeply..."
                    elif elapsed < 60:
                        msg = f"⏱️ {elapsed:.0f}s • Almost there, finalizing..."
                    else:
                        msg = f"⏱️ {elapsed:.0f}s • Complex case detected..."
                    st.caption(msg)
            
            try:
                if use_advanced:
                    update_progress(1, "Detecting language...", 0.1)
                    time.sleep(0.15)
                    
                    update_progress(2, "📋 Extracting entities...", 0.2)
                    extractor = get_entity_extractor()
                    entities = extractor.extract_all(input_text)
                    
                    update_progress(3, "📚 Searching knowledge base...", 0.4)
                    from automotive_intent.agents.orchestrator import get_orchestrator
                    from automotive_intent.agents.state import ChatRequest
                    
                    update_progress(4, "🧠 Multi-agent reasoning...", 0.6)
                    orchestrator = get_orchestrator()
                    response = orchestrator.process_message(ChatRequest(message=input_text.strip()))
                    
                    update_progress(5, "Generating diagnosis...", 0.95)
                    
                    result_dict = {
                        "classification_status": "CONFIRMED" if response.confidence >= 0.7 else "AMBIGUOUS",
                        "intent": response.diagnosis.model_dump() if response.diagnosis else None,
                        "similar_tickets": [t.model_dump() for t in response.similar_tickets],
                        "agent_message": response.message
                    }
                    ticket = None
                else:
                    update_progress(1, "Analyzing input...", 0.2)
                    time.sleep(0.1)
                    
                    update_progress(2, "🧠 LLM classification...", 0.5)
                    request = ClassificationRequest(text=input_text.strip())
                    ticket = st.session_state.pipeline.process(request)
                    result_dict = json.loads(ticket.model_dump_json())
                    
                    update_progress(3, "Complete", 1.0)
                
                progress_placeholder.empty()
                tip_placeholder.empty()
                status_placeholder.empty()
                
            except Exception as e:
                progress_placeholder.empty()
                tip_placeholder.empty()
                status_placeholder.empty()
                st.error(f"Error: {e}")
                return
            
            response_time = time.time() - start_time
            result_dict["mode"] = "advanced" if use_advanced else "fast"
            add_to_history(input_text.strip(), result_dict, response_time, lang)
            
            # Result display
            st.markdown("")
            
            # Header with mode and time
            mode_label = "🧠 Multi-Agent" if use_advanced else "⚡ Fast"
            st.markdown(f"**{mode_label}** • ⏱️ {response_time:.1f}s")
            
            # Status badge
            status = result_dict.get("classification_status", "UNKNOWN")
            status_class = {"CONFIRMED": "status-confirmed", "AMBIGUOUS": "status-ambiguous"}.get(status, "status-out-of-scope")
            
            # Header Row with Status and PDF Button
            col_head, col_action = st.columns([2, 1])
            with col_head:
                st.markdown(f'<span class="{status_class}">{status}</span>', unsafe_allow_html=True)
            
            with col_action:
                # PDF Download Button
                if ticket: # Only available in fast mode for now or if we reconstruct ticket
                    try:
                        pdf_bytes = get_report_generator().generate_job_card(ticket)
                        st.download_button(
                            label="Download Job Card",
                            data=pdf_bytes,
                            file_name=f"job_card_{ticket.request_id[:8]}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.caption("PDF unavailable")
            
            st.markdown("")
            
            if use_advanced:
                # Advanced mode
                if result_dict.get("agent_message"):
                    st.markdown(result_dict["agent_message"])
                
                if result_dict.get("similar_tickets"):
                    st.markdown("---")
                    st.markdown(f"**📋 Similar Cases:** {len(result_dict['similar_tickets'])}")
                    for t in result_dict["similar_tickets"][:3]:
                        st.markdown(f'<div class="ticket-item">• {t.get("complaint", "")[:50]} → <strong>{t.get("system", "")}</strong></div>', unsafe_allow_html=True)
            else:
                # Fast mode
                if ticket and ticket.normalization:
                    st.markdown(f"**{ticket.normalization.technical_summary}**")
                
                if ticket and ticket.intent:
                    st.markdown(f'<div class="intent-path">{ticket.intent.system} → {ticket.intent.component} → {ticket.intent.failure_mode}</div>', unsafe_allow_html=True)
                    
                    # Confidence bar
                    conf_pct = int(ticket.intent.confidence * 100)
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0;">
                        <span style="color: #94a3b8; font-size: 0.85rem;">Confidence:</span>
                        <div class="confidence-bar" style="flex: 1;">
                            <div class="confidence-fill" style="width: {conf_pct}%;"></div>
                        </div>
                        <span style="color: #10B981; font-weight: 700;">{conf_pct}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                if ticket and ticket.triage:
                    st.markdown("")
                    severity_color = {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#eab308", "LOW": "#22c55e"}.get(ticket.triage.severity, "#94a3b8")
                    st.markdown(f'<div class="info-box"><strong style="color: {severity_color}">{ticket.triage.severity}</strong> • {ticket.triage.suggested_action}</div>', unsafe_allow_html=True)
            
            # Entity extraction
            extractor = get_entity_extractor()
            entities = extractor.extract_all(input_text)
            if entities["vehicle"].make or entities["dtc_codes"]:
                st.markdown("---")
                st.markdown("**🔎 Extracted Entities**")
                if entities["vehicle"].make:
                    st.markdown(f"{entities['vehicle'].make} {entities['vehicle'].model or ''} {entities['vehicle'].year or ''}")
                for dtc in entities["dtc_codes"]:
                    st.markdown(f"**{dtc.code}** — {dtc.description}")
        
        elif classify_btn and not input_text.strip():
            st.warning("Please enter a complaint first")
        
        else:
            # Empty state
            st.markdown("""
            <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
                <p style="font-size: 2rem; margin-bottom: 1rem; color: #6b7280;">—</p>
                <p style="font-size: 1.1rem; margin-bottom: 0.5rem;">Enter your vehicle problem</p>
                <p style="font-size: 0.9rem; color: #475569;">Works in <strong>English</strong>, <strong>Hindi</strong>, and <strong>Hinglish</strong></p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
