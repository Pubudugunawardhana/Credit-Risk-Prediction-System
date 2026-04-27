import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# 🌐 PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="Finance & Credit Risk Dashboard",
    page_icon="💼",
    layout="wide",
)

# ----------------------------------------------------------
# 🎨 THEME TOGGLE (Light / Dark)
# ----------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

theme_colors = {
    "dark": {
        "bg": "linear-gradient(135deg, #0A192F 0%, #112240 100%)",  # for CSS
        "bg_plot": "#0A192F",                                     # for matplotlib
        "card": (1, 1, 1, 0.05),
        "text": "#FFFFFF",
        "input_bg": "#1E293B",
        "input_text": "#FFFFFF",
        "accent": "#00BFFF",
        "result": "#1E3A8A",
    },
    "light": {
        "bg": "linear-gradient(135deg, #EAF6FF 0%, #F7FAFC 100%)",  # for CSS
        "bg_plot": "#F7FAFC",                                     # for matplotlib
        "card": (1, 1, 1, 0.9),
        "text": "#000000",
        "input_bg": "#FFFFFF",
        "input_text": "#000000",
        "accent": "#007ACC",
        "result": "#2563EB",
    },
}


current_theme = theme_colors[st.session_state.theme]

# ----------------------------------------------------------
# 💅 CUSTOM STYLING (Dark/Light friendly)
# ----------------------------------------------------------
st.markdown(f"""
    <style>
    body, [data-testid="stAppViewContainer"] {{
        background: {current_theme["bg"]};
        color: {current_theme["text"]} !important;
        font-family: 'Poppins', sans-serif;
    }}

    [data-testid="stSidebar"] {{
        background: {current_theme["card"]};
        backdrop-filter: blur(8px);
        border-right: 1px solid rgba(255,255,255,0.1);
        color: {current_theme["text"]};
    }}

    h1, h2, h3, h4, h5 {{
        color: {current_theme["accent"]} !important;
        font-weight: 600;
    }}

    .step-card {{
        background: {current_theme["card"]};
        border-radius: 15px;
        padding: 20px;
        margin-top: 10px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        color: {current_theme["text"]};
    }}
    .step-card:hover {{
        transform: translateY(-3px);
    }}

    div.stButton > button {{
        background-color: {current_theme["accent"]};
        color: white !important;
        border: none;
        padding: 0.6em 1em;
        border-radius: 10px;
        font-weight: 600;
        transition: 0.3s;
    }}
    div.stButton > button:hover {{
        transform: scale(1.05);
    }}

    /* Input boxes (fixing visibility in both modes) */
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {current_theme["input_bg"]} !important;
        color: {current_theme["input_text"]} !important;
        border-radius: 8px;
    }}

    .stNumberInput label, .stTextInput label, .stSelectbox label {{
        color: {current_theme["text"]} !important;
    }}

    /* Result boxes */
    .result-box {{
        background: {current_theme["result"]};
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-weight: 500;
        margin-top: 10px;
    }}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# 🧭 SIDEBAR NAVIGATION
# ----------------------------------------------------------
st.sidebar.title("🧮 Dashboard Navigation")

# Theme toggle button
theme_label = "🌞 Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
st.sidebar.button(theme_label, on_click=toggle_theme)

app_mode = st.sidebar.radio("Choose an option:", ["🏦 Credit Risk Prediction", "💰 Compound Interest Calculator"])

# ----------------------------------------------------------
# 🧠 LOAD MODEL AND ENCODERS
# ----------------------------------------------------------
model = joblib.load("Extra_tree_credit_model.pkl")
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account"]
}

# ==========================================================
# 🏦 CREDIT RISK PREDICTION
# ==========================================================
if app_mode == "🏦 Credit Risk Prediction":
    st.title("🏦 Credit Risk Prediction App")
    st.write("Predict if a credit applicant’s risk is **Good** or **Bad** using machine learning.")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            sex = st.selectbox("Sex", ["male", "female"])
            job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
            housing = st.selectbox("Housing", ["own", "rent", "free"])
        with col2:
            saving_Accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
            checking_Account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
            credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
            duration = st.number_input("Duration (months)", min_value=1, value=12)

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("🔍 Predict Credit Risk"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [job],
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving_Accounts])[0]],
            "Checking account": [encoders["Checking account"].transform([checking_Account])[0]],
            "Credit amount": [credit_amount],
            "Duration": [duration]
        })

        pred = model.predict(input_df)[0]
        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 1:
            st.markdown('<div class="result-box">✅ The predicted credit risk is: <strong>GOOD</strong></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box" style="background:#B91C1C;">❌ The predicted credit risk is: <strong>BAD</strong></div>', unsafe_allow_html=True)

# ==========================================================
# 💰 COMPOUND INTEREST CALCULATOR + GRAPH
# ==========================================================
elif app_mode == "💰 Compound Interest Calculator":
    st.title("💰 Compound Interest Calculator")
    st.write("Estimate how your investment grows over time through the power of compounding interest.")

    # Step 1
    with st.container():
        st.markdown("<div class='step-card'><h4>Step 1: Initial Investment</h4>", unsafe_allow_html=True)
        principal = st.number_input("Initial Investment ($)", min_value=0.0, value=1000.0, step=100.0)
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 2
    with st.container():
        st.markdown("<div class='step-card'><h4>Step 2: Monthly Contribution</h4>", unsafe_allow_html=True)
        monthly_contrib = st.number_input("Monthly Contribution ($)", value=100.0, step=10.0)
        years = st.number_input("Length of Time (Years)", min_value=1, value=10, step=1)
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 3
    with st.container():
        st.markdown("<div class='step-card'><h4>Step 3: Interest Rate</h4>", unsafe_allow_html=True)
        annual_rate = st.number_input("Estimated Interest Rate (%)", min_value=0.0, value=5.0, step=0.1)
        st.markdown("</div>", unsafe_allow_html=True)

    # Step 4
    with st.container():
        st.markdown("<div class='step-card'><h4>Step 4: Compounding Frequency</h4>", unsafe_allow_html=True)
        freq_options = {"Annually": 1, "Semi-Annually": 2, "Quarterly": 4, "Monthly": 12}
        comp_freq_label = st.selectbox("Compound Frequency", list(freq_options.keys()))
        comp_freq = freq_options[comp_freq_label]
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    calculate = col1.button("🚀 CALCULATE", use_container_width=True)
    reset = col2.button("🔄 RESET", use_container_width=True)

    if calculate:
        r_nominal = annual_rate / 100.0
        n = comp_freq
        t = years
        P = principal
        PMT = monthly_contrib
        EAR = (1 + r_nominal / n) ** n - 1
        monthly_rate = (1 + EAR) ** (1 / 12) - 1
        months = int(t * 12)

        values, balances = [], []
        balance = P
        for m in range(1, months + 1):
            balance = balance * (1 + monthly_rate) + PMT
            values.append(m)
            balances.append(balance)

        future_value = balances[-1]
        total_contributed = P + (PMT * months)
        total_interest = future_value - total_contributed

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box'>📈 <b>Future Value:</b> ${future_value:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box' style='background:#047857;'>💵 <b>Total Contributed:</b> ${total_contributed:,.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result-box' style='background:#92400E;'>✨ <b>Total Interest:</b> ${total_interest:,.2f}</div>", unsafe_allow_html=True)

        # 📊 Graph (Dynamic colors)
        st.markdown("<br><h4>📊 Investment Growth Over Time</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.plot(values, balances, color=current_theme["accent"], linewidth=2)
        ax.set_xlabel("Months", color=current_theme["text"])
        ax.set_ylabel("Balance ($)", color=current_theme["text"])
        ax.tick_params(colors=current_theme["text"])
        ax.grid(True, alpha=0.3)
        ax.set_facecolor(current_theme["card"])
        fig.patch.set_facecolor(current_theme["bg_plot"])
        st.pyplot(fig)
        current_theme["bg_plot"] = "#0A192F"

    if reset:
        st.rerun()
