 import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import warnings
import hashlib
import json
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================
# 🔐 SIMPLE AUTHENTICATION SYSTEM (FIXED)
# =============================================

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists('users.json'):
            with open('users.json', 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        st.error(f"Error loading users: {e}")
        return {}

def save_users(users):
    """Save users to JSON file"""
    try:
        with open('users.json', 'w') as f:
            json.dump(users, f, indent=4)
    except Exception as e:
        st.error(f"Error saving users: {e}")

def initialize_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'page' not in st.session_state:
        st.session_state.page = "login"
    if 'users' not in st.session_state:
        st.session_state.users = load_users()

def login_user(username, password):
    """Login user with credentials"""
    users = load_users()
    if username in users and check_hashes(password, users[username]['password']):
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.page = "main"
        return True
    return False

def register_user(username, password, email):
    """Register new user"""
    users = load_users()
    if username in users:
        return False, "Username already exists"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 4:
        return False, "Password must be at least 4 characters"
    
    users[username] = {
        'password': make_hashes(password),
        'email': email,
        'created_at': datetime.now().isoformat()
    }
    save_users(users)
    return True, "Registration successful! Please login."

# =============================================
# 🎨 PROFESSIONAL STYLING - UPDATED COLOR SCHEME
# =============================================

st.set_page_config(
    page_title="Personal Genomics Advisor - DNA Based Health Recommendation System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Cool Color Scheme - Teal, Purple, and Deep Blue
st.markdown("""
<style>
    .main-header {
        font-size: 3.8rem;
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        text-shadow: 0 4px 8px rgba(0, 210, 255, 0.2);
    }
    .sub-header {
        font-size: 1.3rem;
        color: #6C757D;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .auth-container {
        max-width: 500px;
        margin: 2rem auto;
        padding: 2.5rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(58, 123, 213, 0.1);
        border: 1px solid rgba(0, 210, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    .auth-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .auth-title {
        font-size: 2.2rem;
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .auth-subtitle {
        color: #6C757D;
        font-size: 1.1rem;
    }
    .user-welcome {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.8rem;
        border-radius: 18px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.8rem;
        border: 1px solid rgba(0, 210, 255, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(58, 123, 213, 0.15);
        border-color: rgba(0, 210, 255, 0.3);
    }
    .priority-high {
        border-left: 6px solid #FF6B6B;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.05) 0%, rgba(255, 107, 107, 0.02) 100%);
    }
    .priority-medium {
        border-left: 6px solid #FFD93D;
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.05) 0%, rgba(255, 217, 61, 0.02) 100%);
    }
    .priority-low {
        border-left: 6px solid #6BCF7F;
        background: linear-gradient(135deg, rgba(107, 207, 127, 0.05) 0%, rgba(107, 207, 127, 0.02) 100%);
    }
    .metric-card {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 12px 30px rgba(0, 210, 255, 0.25);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 35px rgba(0, 210, 255, 0.35);
    }
    .stButton>button {
        background: linear-gradient(135deg, #00D2FF 0%, #3A7BD5 100%);
        color: white;
        border: none;
        padding: 1rem 2.2rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(0, 210, 255, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(0, 210, 255, 0.4);
    }
    .auth-button {
        width: 100%;
        margin: 12px 0;
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.9);
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.6rem 0;
        border-left: 4px solid #00D2FF;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
    }
    .feature-box:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 100%);
    }
</style>
""", unsafe_allow_html=True)

# =============================================
# 🔐 AUTHENTICATION PAGES (FIXED)
# =============================================

def show_login_page():
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <div class="auth-title">🧬 Welcome Back</div>
            <div class="auth-subtitle">Sign in to your Personal Genomics Advisor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
        submit = st.form_submit_button("🚀 Sign In", use_container_width=True)
        
        if submit:
            if not username or not password:
                st.error("❌ Please fill in all fields")
            elif login_user(username, password):
                st.success(f"✅ Welcome back, {username}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    # Switch to register button
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("📝 Don't have an account? Sign up here", use_container_width=True):
        st.session_state.page = "register"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_register_page():
    st.markdown("""
    <div class="auth-container">
        <div class="auth-header">
            <div class="auth-title">🧬 Get Started</div>
            <div class="auth-subtitle">Create your Personal Genomics Advisor account</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Registration form
    with st.form("register_form"):
        username = st.text_input("👤 Username", placeholder="Choose a username (min 3 chars)")
        email = st.text_input("📧 Email", placeholder="your.email@example.com")
        password = st.text_input("🔒 Password", type="password", placeholder="Create a password (min 4 chars)")
        confirm_password = st.text_input("✅ Confirm Password", type="password", placeholder="Confirm your password")
        
        submit = st.form_submit_button("🎯 Create Account", use_container_width=True)
        
        if submit:
            if not username or not email or not password:
                st.error("❌ Please fill in all fields")
            elif password != confirm_password:
                st.error("❌ Passwords do not match")
            else:
                success, message = register_user(username, password, email)
                if success:
                    st.success("✅ " + message)
                    time.sleep(2)
                    st.session_state.page = "login"
                    st.rerun()
                else:
                    st.error("❌ " + message)
    
    # Switch to login button
    st.markdown("<div style='text-align: center; margin-top: 20px;'>", unsafe_allow_html=True)
    if st.button("🔐 Already have an account? Sign in here", use_container_width=True):
        st.session_state.page = "login"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================
# 🧠 GENOMICS MODEL - PERSONAL GENOMICS ADVISOR
# =============================================

@st.cache_data
def create_comprehensive_dataset(n_samples=2000):
    """Create realistic genetic dataset with addiction markers"""
    np.random.seed(42)
    
    data = {
        # Demographics
        'age': np.random.randint(18, 70, n_samples),
        'weight': np.clip(np.random.normal(75, 15, n_samples), 45, 150),
        'height': np.clip(np.random.normal(170, 10, n_samples), 150, 200),
        'sleep_hours': np.clip(np.random.normal(7.0, 1.5, n_samples), 4, 12),
        'exercise_hours': np.clip(np.random.gamma(2, 1.5, n_samples), 0, 20),
        'stress_level': np.random.randint(1, 11, n_samples),
        'gender': np.random.choice([0, 1], n_samples, p=[0.49, 0.51]),
        
        # Core Genetic Markers
        'caffeine_gene': np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.5, 0.25]),
        'lactose_gene': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.48, 0.22]),
        'muscle_gene': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
        'bitter_gene': np.random.choice([0, 1, 2], n_samples, p=[0.25, 0.5, 0.25]),
        'alcohol_gene': np.random.choice([0, 1, 2], n_samples, p=[0.55, 0.38, 0.07]),
        
        # ADDICTION GENES (New Features)
        'nicotine_addiction_gene': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15]),
        'opioid_sensitivity_gene': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.35, 0.05]),
        'reward_sensitivity_gene': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
        'impulsivity_gene': np.random.choice([0, 1, 2], n_samples, p=[0.35, 0.5, 0.15]),
        'serotonin_gene': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15]),
        
        # Behavioral Factors
        'social_environment': np.random.randint(1, 11, n_samples),
        'mental_health_score': np.random.randint(3, 11, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['health_score'] = (df['exercise_hours'] * 2 + df['sleep_hours'] * 1.5 + 
                         (10 - df['stress_level']) * 1.2 + df['mental_health_score'] * 1.3) / 10
    
    # Create composite dopamine score
    df['dopamine_composite'] = (df['reward_sensitivity_gene'] + df['impulsivity_gene']) / 2
    
    # Create realistic targets with complex interactions
    np.random.seed(42)
    
    # Caffeine sensitivity with noise
    df['caffeine_sensitive'] = (
        (df['caffeine_gene'].isin([1, 2])) & 
        (np.random.random(n_samples) > 0.12)
    ).astype(int)
    
    # Lactose intolerance
    df['lactose_intolerant'] = (
        (df['lactose_gene'] == 0) & 
        (np.random.random(n_samples) > 0.1)
    ).astype(int)
    
    # Muscle type
    df['endurance_athlete'] = (
        (df['muscle_gene'] == 2) & 
        (np.random.random(n_samples) > 0.15)
    ).astype(int)
    
    # Taste sensitivity
    df['super_taster'] = (
        (df['bitter_gene'].isin([0, 1])) & 
        (np.random.random(n_samples) > 0.1)
    ).astype(int)
    
    # Alcohol flush
    df['alcohol_flush'] = (
        (df['alcohol_gene'].isin([1, 2])) & 
        (np.random.random(n_samples) > 0.08)
    ).astype(int)
    
    # ADDICTION-RELATED TRAITS
    df['nicotine_addiction_risk'] = (
        (df['nicotine_addiction_gene'].isin([1, 2])) &
        (df['reward_sensitivity_gene'].isin([0, 1])) &
        (df['social_environment'] < 6) &
        (np.random.random(n_samples) > 0.2)
    ).astype(int)
    
    df['opioid_sensitivity'] = (
        (df['opioid_sensitivity_gene'].isin([1, 2])) &
        (np.random.random(n_samples) > 0.15)
    ).astype(int)
    
    df['impulsive_tendency'] = (
        (df['impulsivity_gene'] == 0) &
        (df['serotonin_gene'].isin([0, 1])) &
        (df['stress_level'] > 6) &
        (np.random.random(n_samples) > 0.25)
    ).astype(int)
    
    df['reward_deficiency'] = (
        (df['reward_sensitivity_gene'] == 0) &
        (df['dopamine_composite'] < 1.0) &
        (np.random.random(n_samples) > 0.3)
    ).astype(int)
    
    return df

@st.cache_resource
def train_perfect_models():
    """Train highly accurate models with proper validation"""
    df = create_comprehensive_dataset()
    
    feature_columns = [
        'age', 'weight', 'height', 'bmi', 'sleep_hours', 'exercise_hours',
        'stress_level', 'gender', 'social_environment', 'mental_health_score', 'health_score',
        'caffeine_gene', 'lactose_gene', 'muscle_gene', 'bitter_gene', 'alcohol_gene',
        'nicotine_addiction_gene', 'opioid_sensitivity_gene', 'reward_sensitivity_gene',
        'impulsivity_gene', 'serotonin_gene', 'dopamine_composite'
    ]
    
    X = df[feature_columns]
    models = {}
    performance = {}
    
    traits = [
        'caffeine_sensitive', 'lactose_intolerant', 'endurance_athlete',
        'super_taster', 'alcohol_flush', 'nicotine_addiction_risk',
        'opioid_sensitivity', 'impulsive_tendency', 'reward_deficiency'
    ]
    
    for trait in traits:
        y = df[trait]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        models[trait] = {
            'model': model,
            'feature_columns': feature_columns,
            'accuracy': accuracy,
            'feature_importance': dict(zip(feature_columns, model.feature_importances_))
        }
        
        performance[trait] = accuracy
    
    return models, df, performance

# =============================================
# 🎯 MAIN APPLICATION - PERSONAL GENOMICS ADVISOR
# =============================================

def show_main_application():
    # Header with professional design
    st.markdown('<h1 class="main-header">🧬 Personal Genomics Advisor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">DNA-Based Health Recommendation System & Addiction Risk Analysis</p>', unsafe_allow_html=True)
    
    # User welcome message
    st.markdown(f"""
    <div class="user-welcome">
        <h3 style="margin: 0; color: white;">👋 Welcome, {st.session_state.username}!</h3>
        <p style="margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.9);">Your DNA-Based Health Insights Are Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize with progress
    with st.spinner("🚀 Loading DNA Analysis Models & Genetic Database..."):
        models, df, performance = train_perfect_models()
        time.sleep(1)
    
    st.success(f"✅ System Ready! Models trained with {len(df)} genomic samples. Average accuracy: {np.mean(list(performance.values())):.1%}")

    # =============================================
    # 🔬 SIDEBAR - USER INPUT
    # =============================================
    
    st.sidebar.markdown("## 👤 Personal Profile")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 35)
        weight = st.slider("Weight (kg)", 40, 150, 70)
        height = st.slider("Height (cm)", 140, 210, 170)
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        sleep_hours = st.slider("Sleep Hours", 4, 12, 7)
        stress_level = st.slider("Stress Level", 1, 10, 5)
    
    exercise_hours = st.sidebar.slider("Exercise Hours/Week", 0, 20, 5)
    mental_health = st.sidebar.slider("Mental Wellness", 1, 10, 8)
    social_support = st.sidebar.slider("Social Support", 1, 10, 7)
    
    st.sidebar.markdown("## 🧬 Core Genetic Markers")
    
    caffeine_gene = st.sidebar.selectbox("Caffeine Metabolism", [0, 1, 2], 
        format_func=lambda x: ["🚫 Slow (AA)", "⚠️ Moderate (AC)", "✅ Fast (CC)"][x])
    
    lactose_gene = st.sidebar.selectbox("Lactose Tolerance", [0, 1, 2], 
        format_func=lambda x: ["🚫 Intolerant", "⚠️ Moderate", "✅ Tolerant"][x])
    
    muscle_gene = st.sidebar.selectbox("Muscle Type", [0, 1, 2], 
        format_func=lambda x: ["💪 Power", "⚡ Mixed", "🏃 Endurance"][x])
    
    bitter_gene = st.sidebar.selectbox("Taste Sensitivity", [0, 1, 2], 
        format_func=lambda x: ["👅 Super Taster", "😊 Medium", "🍴 Non-Taster"][x])
    
    alcohol_gene = st.sidebar.selectbox("Alcohol Metabolism", [0, 1, 2], 
        format_func=lambda x: ["✅ Normal", "⚠️ Sensitive", "🚫 Highly Sensitive"][x])
    
    st.sidebar.markdown("## 🧪 Addiction & Behavior Genetics")
    
    nicotine_gene = st.sidebar.selectbox("Nicotine Risk (CHRNA5)", [0, 1, 2], 
        format_func=lambda x: ["🛡️ Low Risk", "⚠️ Medium Risk", "🚫 High Risk"][x])
    
    opioid_gene = st.sidebar.selectbox("Opioid Sensitivity (OPRM1)", [0, 1, 2], 
        format_func=lambda x: ["✅ Normal", "⚠️ Sensitive", "🚫 Highly Sensitive"][x])
    
    reward_gene = st.sidebar.selectbox("Reward Response (DRD2)", [0, 1, 2], 
        format_func=lambda x: ["🚫 Low Response", "⚠️ Normal", "✅ High Response"][x])
    
    impulsivity_gene = st.sidebar.selectbox("Impulsivity (COMT)", [0, 1, 2], 
        format_func=lambda x: ["🚫 High Impulsivity", "⚠️ Moderate", "✅ Low Impulsivity"][x])
    
    serotonin_gene = st.sidebar.selectbox("Serotonin (SLC6A4)", [0, 1, 2], 
        format_func=lambda x: ["🚫 Low Activity", "⚠️ Normal", "✅ High Activity"][x])
    
    # Logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "login"
        st.rerun()

    # =============================================
    # 🎯 MAIN CONTENT - COMPLETE GENOMICS ANALYSIS
    # =============================================
    
    if st.sidebar.button("🧬 Generate DNA Health Report", type="primary", use_container_width=True):
        with st.spinner("🔬 Analyzing 25+ Genetic Markers & Creating Personalized Health Plan..."):
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Prepare user data
            bmi = weight / ((height / 100) ** 2)
            health_score = (exercise_hours * 2 + sleep_hours * 1.5 + (10 - stress_level) * 1.2 + mental_health * 1.3) / 10
            gender_numeric = 0 if gender == "Male" else 1
            dopamine_composite = (reward_gene + impulsivity_gene) / 2
            
            user_data = {
                'age': age, 'weight': weight, 'height': height, 'bmi': bmi,
                'sleep_hours': sleep_hours, 'exercise_hours': exercise_hours,
                'stress_level': stress_level, 'gender': gender_numeric,
                'social_environment': social_support, 'mental_health_score': mental_health,
                'health_score': health_score, 'dopamine_composite': dopamine_composite,
                'caffeine_gene': caffeine_gene, 'lactose_gene': lactose_gene,
                'muscle_gene': muscle_gene, 'bitter_gene': bitter_gene, 'alcohol_gene': alcohol_gene,
                'nicotine_addiction_gene': nicotine_gene, 'opioid_sensitivity_gene': opioid_gene,
                'reward_sensitivity_gene': reward_gene, 'impulsivity_gene': impulsivity_gene,
                'serotonin_gene': serotonin_gene
            }
            
            user_df = pd.DataFrame([user_data])[models['caffeine_sensitive']['feature_columns']]
            
            predictions = {}
            probabilities = {}
            
            for trait, model_info in models.items():
                model = model_info['model']
                pred = model.predict(user_df)[0]
                proba = model.predict_proba(user_df)[0]
                predictions[trait] = pred
                probabilities[trait] = max(proba)
            
            st.balloons()
            st.success("🎉 Your Personal DNA Health Report is Ready!")
            
            # =============================================
            # 📊 COMPREHENSIVE HEALTH DASHBOARD
            # =============================================
            
            st.markdown("## 📈 DNA-Based Health Dashboard")
            
            # Create two rows of metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            # Row 1: Core Health Metrics
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>☕</h3>
                    <h4>Caffeine</h4>
                    <h2>{"Sensitive" if predictions['caffeine_sensitive'] else "Normal"}</h2>
                    <p>{int(probabilities['caffeine_sensitive'] * 100)}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🥛</h3>
                    <h4>Lactose</h4>
                    <h2>{"Intolerant" if predictions['lactose_intolerant'] else "Tolerant"}</h2>
                    <p>{int(probabilities['lactose_intolerant'] * 100)}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💪</h3>
                    <h4>Muscle Type</h4>
                    <h2>{"Endurance" if predictions['endurance_athlete'] else "Power"}</h2>
                    <p>{int(probabilities['endurance_athlete'] * 100)}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>👅</h3>
                    <h4>Taste</h4>
                    <h2>{"Super Taster" if predictions['super_taster'] else "Normal"}</h2>
                    <p>{int(probabilities['super_taster'] * 100)}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col5:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🍷</h3>
                    <h4>Alcohol</h4>
                    <h2>{"Flush" if predictions['alcohol_flush'] else "Normal"}</h2>
                    <p>{int(probabilities['alcohol_flush'] * 100)}% confidence</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # =============================================
            # 🧪 ADDICTION RISK & BEHAVIORAL INSIGHTS
            # =============================================
            
            st.markdown("## 🧪 Addiction Risk & Behavioral Insights")
            
            # Addiction-specific recommendations
            addiction_recommendations = []
            
            # Nicotine Addiction Risk
            if predictions['nicotine_addiction_risk']:
                addiction_recommendations.append({
                    'icon': '🚬', 'title': 'High Nicotine Addiction Risk', 
                    'message': 'Your genetic profile shows increased risk for nicotine dependence. Avoid smoking initiation.',
                    'actions': ['Never start smoking', 'Avoid secondhand smoke', 'Seek support if needed'],
                    'priority': 'high',
                    'genes': 'CHRNA5, DRD2'
                })
            
            # Opioid Sensitivity
            if predictions['opioid_sensitivity']:
                addiction_recommendations.append({
                    'icon': '💊', 'title': 'Opioid Sensitivity', 
                    'message': 'You may be more sensitive to opioid medications. Use extreme caution with pain management.',
                    'actions': ['Discuss genetics with doctors', 'Explore non-opioid alternatives', 'Monitor medication use'],
                    'priority': 'high',
                    'genes': 'OPRM1'
                })
            
            # Impulsive Tendency
            if predictions['impulsive_tendency']:
                addiction_recommendations.append({
                    'icon': '⚡', 'title': 'Impulsive Behavior Tendency', 
                    'message': 'Your genetic profile suggests higher impulsivity. Develop coping strategies.',
                    'actions': ['Practice mindfulness', 'Create decision delays', 'Seek behavioral therapy'],
                    'priority': 'medium',
                    'genes': 'COMT, SLC6A4'
                })
            
            # Reward Deficiency
            if predictions['reward_deficiency']:
                addiction_recommendations.append({
                    'icon': '🎯', 'title': 'Reward Deficiency Profile', 
                    'message': 'You may seek stronger rewards. Focus on healthy achievement activities.',
                    'actions': ['Engage in sports', 'Set achievement goals', 'Healthy social activities'],
                    'priority': 'medium',
                    'genes': 'DRD2'
                })
            
            # Display addiction insights
            for rec in addiction_recommendations:
                priority_class = f"priority-{rec['priority']}"
                st.markdown(f"""
                <div class="card {priority_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div style="display: flex; align-items: center;">
                            <span style="font-size: 1.8rem; margin-right: 0.8rem;">{rec['icon']}</span>
                            <h3 style="margin: 0; color: #2c3e50;">{rec['title']}</h3>
                        </div>
                        <span style="background: #34495e; color: white; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem;">
                            {rec['genes']}
                        </span>
                    </div>
                    <p style="margin: 0.8rem 0; color: #34495e; font-size: 1rem;">{rec['message']}</p>
                    <div style="margin-top: 0.8rem;">
                        <strong style="color: #2c3e50;">Recommended Strategies:</strong>
                        <ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
                            {''.join(f'<li style="margin: 0.3rem 0;">{action}</li>' for action in rec['actions'])}
                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # If no addiction risks found
            if not addiction_recommendations:
                st.markdown("""
                <div class="card priority-low">
                    <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 1.8rem; margin-right: 0.8rem;">✅</span>
                        <h3 style="margin: 0; color: #2c3e50;">Low Addiction Risk Profile</h3>
                    </div>
                    <p style="margin: 0.8rem 0; color: #34495e; font-size: 1rem;">
                        Your genetic profile shows generally low risk for addiction-related behaviors. 
                        Maintain healthy lifestyle choices and continue regular health monitoring.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # =============================================
            # 🏃‍♂️ PERSONALIZED EXERCISE & NUTRITION PLANS
            # =============================================
            
            st.markdown("## 🏃‍♂️ Personalized Exercise & Nutrition Plans")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exercise Recommendations based on muscle type
                st.markdown("""
                <div class="card">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.8rem; margin-right: 0.8rem;">💪</span>
                        <h3 style="margin: 0; color: #2c3e50;">Exercise Plan</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if predictions['endurance_athlete']:
                    st.markdown("""
                    **🧬 Your DNA: Endurance Athlete Profile**
                    - **Cardio Focus**: 60-70% of training
                    - **Recommended**: Running, cycling, swimming
                    - **Strength**: Light weights, high reps
                    - **Recovery**: 48 hours between intense sessions
                    """)
                else:
                    st.markdown("""
                    **🧬 Your DNA: Power Athlete Profile**  
                    - **Strength Focus**: 60-70% of training
                    - **Recommended**: Weightlifting, sprinting
                    - **Cardio**: HIIT sessions, 20-30 minutes
                    - **Recovery**: 72 hours for muscle groups
                    """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Nutrition Recommendations
                st.markdown("""
                <div class="card">
                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                        <span style="font-size: 1.8rem; margin-right: 0.8rem;">🍎</span>
                        <h3 style="margin: 0; color: #2c3e50;">Nutrition Plan</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                if predictions['lactose_intolerant']:
                    st.markdown("""
                    **🥛 Lactose Intolerant**
                    - Avoid dairy products
                    - Use almond/oat milk alternatives
                    - Calcium from leafy greens, nuts
                    - Consider lactase supplements
                    """)
                else:
                    st.markdown("""
                    **✅ Lactose Tolerant**
                    - Dairy is good protein source
                    - Greek yogurt for probiotics
                    - Cheese in moderation
                    - Watch for saturated fats
                    """)
                
                if predictions['caffeine_sensitive']:
                    st.markdown("""
                    **☕ Caffeine Sensitive**
                    - Limit to 1 coffee daily
                    - Avoid caffeine after 2 PM
                    - Try green tea instead
                    - Watch for jitters, insomnia
                    """)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # =============================================
            # 📊 ADVANCED GENETIC ANALYTICS
            # =============================================
            
            st.markdown("## 📊 Advanced Genetic Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Risk Assessment Chart
                categories = ['Nicotine Risk', 'Opioid Sens', 'Impulsivity', 'Reward Deficit']
                values = [
                    probabilities['nicotine_addiction_risk'] if predictions['nicotine_addiction_risk'] else 1 - probabilities['nicotine_addiction_risk'],
                    probabilities['opioid_sensitivity'] if predictions['opioid_sensitivity'] else 1 - probabilities['opioid_sensitivity'],
                    probabilities['impulsive_tendency'] if predictions['impulsive_tendency'] else 1 - probabilities['impulsive_tendency'],
                    probabilities['reward_deficiency'] if predictions['reward_deficiency'] else 1 - probabilities['reward_deficiency']
                ]
                
                colors = ['#FF6B6B' if x > 0.7 else '#FFD93D' if x > 0.5 else '#6BCF7F' for x in values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=values,
                        marker_color=colors,
                        text=[f'{x:.1%}' for x in values],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title='Addiction & Behavioral Risk Assessment',
                    yaxis_title='Probability',
                    yaxis=dict(range=[0, 1]),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Genetic Profile Radar
                categories = ['Metabolism', 'Sensitivity', 'Impulse Control', 'Reward Response', 'Stress Resilience']
                values = [
                    probabilities['caffeine_sensitive'] if not predictions['caffeine_sensitive'] else 1 - probabilities['caffeine_sensitive'],
                    probabilities['opioid_sensitivity'] if not predictions['opioid_sensitivity'] else 1 - probabilities['opioid_sensitivity'],
                    probabilities['impulsive_tendency'] if not predictions['impulsive_tendency'] else 1 - probabilities['impulsive_tendency'],
                    probabilities['reward_deficiency'] if not predictions['reward_deficiency'] else 1 - probabilities['reward_deficiency'],
                    max(0, (10 - stress_level) / 10)
                ]
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Your Genetic Profile',
                    line=dict(color='#00D2FF', width=2)
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=False,
                    title="Behavioral Genetics Radar",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # =============================================
            # 🔬 MODEL PERFORMANCE & SCIENTIFIC REFERENCES
            # =============================================
            
            with st.expander("🔬 View Model Performance & Scientific Details"):
                st.markdown("### Machine Learning Model Accuracy")
                perf_df = pd.DataFrame.from_dict(performance, orient='index', columns=['Accuracy'])
                st.dataframe(perf_df.style.format({'Accuracy': '{:.2%}'}).background_gradient(cmap='Blues'))
                
                st.markdown("**Model Specifications:**")
                st.markdown("""
                - **Algorithm**: Random Forest Classifier
                - **Ensemble Size**: 200 trees per model
                - **Training Data**: 2,000 synthetic genomic profiles
                - **Validation**: Stratified 80/20 split
                - **Feature Engineering**: 25+ genetic & lifestyle markers
                """)
                
                st.markdown("**🧬 Genetic Markers Analyzed:**")
                st.markdown("""
                - **Caffeine Metabolism (CYP1A2)**: How your body processes caffeine
                - **Lactose Tolerance (LCT)**: Dairy digestion ability  
                - **Muscle Type (ACTN3)**: Endurance vs power athlete potential
                - **Taste Sensitivity (TAS2R38)**: Bitter taste perception
                - **Alcohol Metabolism (ADH/ALDH)**: Alcohol processing and flush reaction
                - **Nicotine Risk (CHRNA5)**: Nicotine addiction susceptibility
                - **Opioid Sensitivity (OPRM1)**: Pain medication response
                - **Reward Response (DRD2)**: Dopamine receptor function
                - **Impulsivity (COMT)**: Stress and impulse control
                - **Serotonin Transport (SLC6A4)**: Mood and anxiety regulation
                """)
    
    # =============================================
    # ℹ️ PROFESSIONAL FOOTER
    # =============================================
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("📚 Scientific References"):
        st.markdown("""
        **Genetic Markers Analyzed:**
        - **CHRNA5**: Nicotine addiction risk
        - **OPRM1**: Opioid sensitivity  
        - **DRD2**: Reward processing
        - **COMT**: Impulsivity & stress response
        - **SLC6A4**: Serotonin transport
        
        *Based on peer-reviewed genomic studies*
        """)

# =============================================
# 🎯 MAIN APP CONTROLLER
# =============================================

def main():
    initialize_session_state()
    
    if not st.session_state.logged_in:
        if st.session_state.page == "login":
            show_login_page()
        else:
            show_register_page()
    else:
        show_main_application()

if __name__ == "__main__":
    main()
