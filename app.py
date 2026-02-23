"""
DOCWISE AI - Unified Medical Application
Combines PDF Summarization and Doctor Recommendation System
"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import time
from pathlib import Path
import sys

# Add modules to path
sys.path.append(str(Path(__file__).parent))

# Import modules
from modules.disease_mapper import predict_specialist
from modules.doctor_filtering import get_doctors_by_specialist

# For PDF summarization
from transformers import BartForConditionalGeneration, BartTokenizer
import PyPDF2
import io

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="DOCWISE AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Primary color scheme */
    :root {
        --primary-color: #1193d4;
        --background-light: #f6f7f8;
        --background-dark: #101c22;
    }
    
    /* Main container */
    .main {
        background-color: var(--background-light);
    }
    
    /* Header styling */
    .docwise-header {
        background: linear-gradient(135deg, #1193d4 0%, #0e7ab8 100%);
        padding: 2rem;
        border-radius: 0.75rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .docwise-title {
        font-size: 3rem;
        font-weight: 900;
        color: white;
        letter-spacing: 0.1em;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .docwise-subtitle {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        margin-top: 0.5rem;
    }
    
    /* Dashboard cards */
    .dashboard-card {
        background: white;
        border-radius: 0.75rem;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1.5rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(17, 147, 212, 0.2);
    }
    
    /* Upload area */
    .upload-area {
        border: 2px dashed rgba(17, 147, 212, 0.5);
        border-radius: 0.75rem;
        padding: 3rem;
        text-align: center;
        background: rgba(17, 147, 212, 0.05);
        margin: 1.5rem 0;
    }
    
    /* Summary box */
    .summary-box {
        background: linear-gradient(to right, #f8f9fa, #ffffff);
        border-left: 4px solid #1193d4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .summary-box p {
        color: #000000 !important;  /* Ensure summary text is visible */
    }
    
    /* Doctor card */
    .doctor-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }
    
    .doctor-card:hover {
        box-shadow: 0 4px 16px rgba(17, 147, 212, 0.3);
        border-color: #1193d4;
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #1193d4 0%, #0e7ab8 100%);
        color: white;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1193d4 0%, #0e7ab8 100%);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0e7ab8 0%, #0a5a8a 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(17, 147, 212, 0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1193d4 0%, #0a5a8a 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ============ LOAD MODELS (CACHED) ============
@st.cache_resource
def load_bart_model():
    """Load BART model for PDF summarization"""
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

@st.cache_resource
def load_doctor_data():
    """Load doctor profiles CSV"""
    try:
        df = pd.read_csv("data/doctor_profiles.csv")
        return df
    except:
        return None

# ============ PDF FUNCTIONS ============
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

def generate_summary(text, tokenizer, model, max_length=200, min_length=50):
    """Generate summary using BART model - optimized for speed"""
    try:
        # Truncate text for faster processing
        inputs = tokenizer.encode(
            "summarize: " + text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        )
        
        # Generate summary with optimized parameters for speed
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=2,  # Reduced from 4 for speed
            length_penalty=1.5,  # Reduced from 2.0
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        summary = tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )
        
        return summary
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# ============ DOCTOR DASHBOARD ============
def doctor_dashboard():
    """Doctor Dashboard - PDF Summarization"""
    st.markdown("""
    <div class="docwise-header">
        <h1 class="docwise-title">👨‍⚕️ Doctor Dashboard</h1>
        <p class="docwise-subtitle">PDF Medical Report Summarizer</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load BART model
    with st.spinner("Loading AI model..."):
        tokenizer, model = load_bart_model()
    
    st.success("✅ AI Model loaded successfully")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📄 Upload PDF Report")
        st.markdown("""
        <div class="upload-area">
            <h3>📤 Drag and drop or browse to upload a PDF</h3>
            <p style="color: #666;">Upload a medical report to generate an automatic summary</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_pdf = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a text-based medical PDF report",
            label_visibility="collapsed"
        )
        
        # Summarization parameters
        with st.expander("⚙️ Summarization Settings"):
            max_length = st.slider("Maximum Summary Length", 50, 5000, 200, 10)
            min_length = st.slider("Minimum Summary Length", 10, 500, 50, 5)
    
    with col2:
        st.markdown("### 📊 Generated Summary")
        
        if uploaded_pdf is not None:
            # Show file info
            st.info(f"📎 File: {uploaded_pdf.name} ({uploaded_pdf.size / 1024:.2f} KB)")
            
            if st.button("🚀 Generate Summary", use_container_width=True):
                # Start timer
                start_time = time.time()
                
                with st.spinner("🔄 Extracting text from PDF..."):
                    try:
                        final_text = extract_text_from_pdf(uploaded_pdf)
                        word_count = len(final_text.split())
                        st.caption(f"Words detected: {word_count}")
                    except Exception as e:
                        st.error(f"❌ Error reading PDF: {str(e)}")
                        return
                
                with st.spinner("🤖 Generating AI summary..."):
                    try:
                        summary = generate_summary(
                            final_text,
                            tokenizer,
                            model,
                            max_length,
                            min_length
                        )
                        
                        # End timer
                        end_time = time.time()
                        processing_time = end_time - start_time
                        
                        # Display summary
                        st.markdown(f"""
                        <div class="summary-box">
                            <h4>📄 Summary</h4>
                            <p>{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        summary_words = len(summary.split())
                        compression = round((1 - summary_words / word_count) * 100, 1)
                        
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{word_count}</div>
                                <div class="metric-label">Original Words</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{summary_words}</div>
                                <div class="metric-label">Summary Words</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_c:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{compression}%</div>
                                <div class="metric-label">Compression</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Processing time
                        st.success(f"⏱️ Processing completed in {processing_time:.2f} seconds")
                        
                        # Download button
                        st.download_button(
                            "📥 Download Summary",
                            summary,
                            file_name="medical_summary.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {str(e)}")
        else:
            st.markdown("""
            <div class="summary-box">
                <p style="color: #666; text-align: center;">
                    No summary generated yet. Please upload a PDF report to view the summary here.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ============ PATIENT DASHBOARD ============
def patient_dashboard():
    """Patient Dashboard - Doctor Recommendation"""
    st.markdown("""
    <div class="docwise-header">
        <h1 class="docwise-title">🧑‍🤝‍🧑 Patient Dashboard</h1>
        <p class="docwise-subtitle">Find a Doctor, Instantly</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Search section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🔍 Find Your Doctor")
        
        # Input fields
        disease = st.text_input(
            "🏥 Enter your symptoms or disease",
            placeholder="e.g., diabetes, headache, fever",
            help="Enter the condition or symptoms you're experiencing"
        )
        
        location = st.text_input(
            "📍 Enter your location",
            placeholder="e.g., Chennai, Mumbai, Delhi",
            help="Enter your preferred location for doctor search"
        )
        
        search_clicked = st.button("🔎 Find Doctors", use_container_width=True)
    
    with col2:
        st.markdown("### 💡 How It Works")
        st.info("""
        1️⃣ Enter your symptoms or disease name
        
        2️⃣ Specify your preferred location
        
        3️⃣ Get instant recommendations for top-rated specialists
        
        4️⃣ View doctor profiles with experience and ratings
        """)
    
    # Results section
    if search_clicked and disease:
        st.markdown("---")
        st.markdown("### 🩺 Search Results")
        
        with st.spinner("🔍 Finding the best doctors for you..."):
            # Predict specialist
            specialist = predict_specialist(disease)
            
            if specialist:
                st.success(f"✅ Recommended Specialist: **{specialist}**")
                
                # Get doctors
                try:
                    doctors_df = get_doctors_by_specialist(
                        specialist, 
                        location=location if location else None,
                        min_experience=2,
                        min_rating=3.5
                    )
                    
                    if not doctors_df.empty:
                        # Sort by rating
                        doctors_df = doctors_df.sort_values(by="Rating", ascending=False)
                        
                        st.markdown(f"### 👨‍⚕️ Top {len(doctors_df)} Doctors Found")
                        
                        # Display doctor cards
                        for idx, (_, doctor) in enumerate(doctors_df.iterrows(), 1):
                            st.markdown(f"""
                            <div class="doctor-card">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <h3 style="margin: 0; color: #1193d4;">#{idx} {doctor['Name']}</h3>
                                        <p style="margin: 5px 0; color: #666;">
                                            👨‍⚕️ {doctor['Specialist']} | 
                                            💼 {doctor['Experience']} years experience | 
                                            ⭐ {doctor['Rating']}/5.0
                                        </p>
                                        <p style="margin: 5px 0; color: #666;">
                                            🏢 {doctor['Location']} | 
                                            📞 {doctor['Contact']}
                                        </p>
                                    </div>
                                    <div style="text-align: center; padding: 1rem;">
                                        <div style="background: linear-gradient(135deg, #1193d4 0%, #0e7ab8 100%); 
                                                    color: white; 
                                                    border-radius: 50%; 
                                                    width: 60px; 
                                                    height: 60px; 
                                                    display: flex; 
                                                    align-items: center; 
                                                    justify-content: center;
                                                    font-size: 1.5rem;
                                                    font-weight: bold;">
                                            {doctor['Rating']}
                                        </div>
                                        <p style="margin: 5px 0; font-size: 0.8rem;">Rating</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                    else:
                        st.warning("⚠️ No suitable doctors found in your area. Try expanding your search location.")
                        
                except Exception as e:
                    st.error(f"❌ Error fetching doctors: {str(e)}")
            else:
                st.error("❌ Disease not found in our database. Please try a different search term.")
    
    elif search_clicked and not disease:
        st.warning("⚠️ Please enter a disease or symptom to search for doctors.")

# ============ MAIN APP ============
def main():
    """Main application - no authentication required"""
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: white; font-weight: 900; letter-spacing: 0.1em;">DOCWISE AI</h2>
            <p style="color: rgba(255,255,255,0.8);">Medical Intelligence Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation menu
        selected = option_menu(
            menu_title="Dashboard",
            options=["👨‍⚕️ Doctor", "🧑‍🤝‍🧑 Patient"],
            icons=["hospital", "people"],
            default_index=0,
            orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "transparent"},
                "icon": {"color": "white", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "5px",
                    "--hover-color": "rgba(255,255,255,0.2)",
                    "color": "white",
                },
                "nav-link-selected": {"background-color": "rgba(255,255,255,0.3)", "font-weight": "bold"},
            }
        )
        
        st.markdown("---")
        st.markdown("<p style='color: white; text-align: center;'>DOCWISE AI © 2025</p>", unsafe_allow_html=True)
    
    # Route to selected dashboard
    if selected == "👨‍⚕️ Doctor":
        doctor_dashboard()
    else:
        patient_dashboard()

if __name__ == "__main__":
    main()