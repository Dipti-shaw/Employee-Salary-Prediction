
import streamlit as st
import pandas as pd
import joblib

# Attempt to import Plotly with error handling
try:
    import plotly.express as px # type: ignore
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Salary range visualization is unavailable. Please install Plotly using 'pip install plotly'.")

# Attempt to load trained pipeline with error handling
try:
    model = joblib.load("salary_pipeline.pkl")
except FileNotFoundError:
    model = None
    st.error("Model file 'salary_pipeline.pkl' not found. Please ensure the model file is in the correct directory or contact support.")
except Exception as e:
    model = None
    st.error(f"Error loading model: {str(e)}. Please contact support.")

# Set page config to remove top whitespace and set dark theme
st.set_page_config(
    page_title="Employee Salary Predictor",
    layout="centered",
    page_icon="💼",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a professional dark-themed UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Poppins', sans-serif;
    }
    body {
        background-color: #1a1a1a;
        color: #ffffff;
        margin: 0;
        padding: 0;
    }
    .main-container {
        background: #2c2c2c;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        margin: 0 auto;
        max-width: 900px;
    }
    h1 {
        color: #00bcd4;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .subtitle {
        text-align: center;
        color: #b0bec5;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00bcd4, #0097a7);
        color: white;
        border-radius: 25px;
        font-weight: 600;
        padding: 12px 30px;
        font-size: 1.1rem;
        border: none;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0097a7, #006d77);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .stSelectbox, .stSlider {
        background-color: #3a3a3a;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 8px;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    .stSlider > div > div > div {
        background-color: #00bcd4;
    }
    .result-box {
        background: #3a3a3a;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 6px solid #00bcd4;
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        margin-top: 2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .info-box {
        background-color: #263238;
        padding: 1rem;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #b0bec5;
        margin-bottom: 1rem;
        border-left: 4px solid #00bcd4;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #444;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-container {
        animation: fadeIn 0.8s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Main container
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1>SalarySync Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Discover your earning potential with our AI-powered salary estimation tool.</p>", unsafe_allow_html=True)

    # Input form with guidance
    st.markdown("<div class='info-box'>Select your profile details to get an accurate salary prediction based on industry trends.</div>", unsafe_allow_html=True)
    with st.form("salary_form"):
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            experience = st.slider("Years of Experience", 0, 20, 5, help="Select your total years of professional experience.")
            education = st.selectbox("Education Level", ["B.Tech", "M.Tech", "MBA", "PhD"], help="Choose your highest education qualification.")
            department = st.selectbox("Department", ["IT", "Data", "HR", "Product", "Marketing", "R&D", "Business", "Operations", "Finance"], help="Select the department you work in.")
        with col2:
            job_title = st.selectbox("Job Title", ["Software Engineer", "Data Scientist", "Backend Developer", "HR Manager", "Product Manager", "DevOps Engineer", "Frontend Developer", "Marketing Lead", "Research Scientist", "Business Analyst"], help="Choose your current or desired job role.")
            location = st.selectbox("Location", ["Delhi", "Bangalore", "Mumbai", "Chennai", "Hyderabad", "Pune", "Kolkata"], help="Select the city where you work or plan to work.")

        submitted = st.form_submit_button("Predict My Salary")

    # Prediction logic with visualization
    if submitted:
        if model is None:
            st.warning("Cannot generate prediction because the model is unavailable. Please try again later or contact support.")
        else:
            try:
                input_df = pd.DataFrame([{
                    "Experience": experience,
                    "Education_Level": education,
                    "Job_Title": job_title,
                    "Location": location,
                    "Department": department
                }])

                salary = model.predict(input_df)[0]
                st.markdown("###  Your Estimated Salary")
                st.markdown(f"<div class='result-box'>₹{int(salary):,}</div>", unsafe_allow_html=True)

                # Salary range visualization
                salary_range = pd.DataFrame({
                    "Category": ["Predicted Salary", "Average Salary", "Max Salary"],
                    "Amount": [salary, salary * 0.9, salary * 1.2]  # Mock range for visualization
                })

                if PLOTLY_AVAILABLE:
                    try:
                        fig = px.bar(
                            salary_range,
                            x="Category",
                            y="Amount",
                            title="Salary Range Comparison",
                            color="Category",
                            color_discrete_sequence=["#00bcd4", "#b0bec5", "#ff6f61"],
                            height=350
                        )
                        fig.update_layout(
                            xaxis_title="",
                            yaxis_title="Salary (₹)",
                            showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#ffffff"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Failed to generate visualization: {str(e)}. Displaying salary range as text instead.")
                        st.markdown("#### Salary Range")
                        st.write(f"- Predicted Salary: ₹{int(salary):,}")
                        st.write(f"- Average Salary: ₹{int(salary * 0.9):,}")
                        st.write(f"- Max Salary: ₹{int(salary * 1.2):,}")
                else:
                    st.markdown("#### Salary Range")
                    st.write(f"- Predicted Salary: ₹{int(salary):,}")
                    st.write(f"- Average Salary: ₹{int(salary * 0.9):,}")
                    st.write(f"- Max Salary: ₹{int(salary * 1.2):,}")

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}. Please check your inputs or contact support.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
        © 2025 SalarySync Predictor • Powered by Streamlit • <a href='https://x.ai' style='color: #00bcd4; text-decoration: none;'>Learn More</a>
    </div>
""", unsafe_allow_html=True)