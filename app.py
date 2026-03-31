
import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =========================
# PDF libraries
# =========================
PDF_PLUMBER_AVAILABLE = False
PYMUPDF_AVAILABLE = False
PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDF_PLUMBER_AVAILABLE = True
except Exception:
    pass

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    pass

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    pass

# =========================
# Page settings
# =========================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Paths
# =========================
MODEL_DIR = "resume_job_model_artifacts_v2"
JOB_CATALOG_PATH = os.path.join(MODEL_DIR, "job_catalog.csv")
JOB_EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "job_embeddings.npy")
SKILL_LEXICON_PATH = os.path.join(MODEL_DIR, "skill_lexicon.json")
MODEL_CONFIG_PATH = os.path.join(MODEL_DIR, "model_config.json")

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 17px;
    color: #9aa4b2;
    margin-bottom: 1.2rem;
}
.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    text-align: center;
    margin-bottom: 12px;
}
.metric-title {
    font-size: 15px;
    color: #cbd5e1;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white;
}
.section-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
}
.good-box {
    padding: 14px;
    border-radius: 14px;
    background: rgba(34,197,94,0.15);
    border-left: 6px solid #22c55e;
    color: #bbf7d0;
    font-weight: 600;
}
.mid-box {
    padding: 14px;
    border-radius: 14px;
    background: rgba(245,158,11,0.15);
    border-left: 6px solid #f59e0b;
    color: #fde68a;
    font-weight: 600;
}
.bad-box {
    padding: 14px;
    border-radius: 14px;
    background: rgba(239,68,68,0.15);
    border-left: 6px solid #ef4444;
    color: #fecaca;
    font-weight: 600;
}
.tag {
    display: inline-block;
    padding: 8px 12px;
    margin: 5px 6px 5px 0;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
}
.tag-found {
    background: rgba(34,197,94,0.15);
    color: #bbf7d0;
    border: 1px solid rgba(34,197,94,0.35);
}
.tag-missing {
    background: rgba(239,68,68,0.15);
    color: #fecaca;
    border: 1px solid rgba(239,68,68,0.35);
}
.small-note {
    font-size: 13px;
    color: #9aa4b2;
}
.footer-note {
    color: #94a3b8;
    font-size: 13px;
}
.preview-box {
    padding: 12px;
    border-radius: 12px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    max-height: 180px;
    overflow-y: auto;
    font-size: 13px;
    color: #d1d5db;
    white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Utilities
# =========================
def clean_text(text):
    if text is None:
        return ""
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+#./ -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def split_skills(skills_text):
    if skills_text is None:
        return []
    text = str(skills_text)
    parts = re.split(r"[,;/|\n]", text)
    parts = [p.strip().lower() for p in parts if p.strip()]
    parts = [re.sub(r"\s+", " ", p) for p in parts]
    parts = [p for p in parts if 1 < len(p) <= 60]
    return sorted(list(set(parts)))

def normalize_skill_token(skill):
    return re.sub(r"\s+", " ", str(skill).lower().strip())

def extract_skills_from_text(text, skill_lexicon):
    text = " " + clean_text(text) + " "
    found = set()
    for skill in skill_lexicon:
        pattern = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
        if re.search(pattern, text):
            found.add(skill)
    return sorted(found)

def semantic_score_from_cosine(cos_sim):
    score = (cos_sim - 0.20) / (0.90 - 0.20)
    score = max(0.0, min(1.0, score))
    return score * 100.0

def hybrid_final_score(semantic_score, skill_score, missing_count):
    final_score = 0.45 * semantic_score + 0.55 * skill_score

    if skill_score == 100 and semantic_score >= 65:
        final_score += 10
    elif skill_score >= 80 and semantic_score >= 55:
        final_score += 5

    if missing_count >= 5:
        final_score -= 10
    elif missing_count >= 3:
        final_score -= 5

    final_score = max(0.0, min(100.0, final_score))
    return round(final_score, 2)

def build_recommendation(score, missing_skills_count):
    if score >= 85 and missing_skills_count <= 1:
        return "Excellent match for the role."
    elif score >= 75 and missing_skills_count <= 2:
        return "Strong match for the role."
    elif score >= 60:
        return "Moderate match. Candidate is suitable but has some skill gaps."
    else:
        return "Weak match. Candidate needs significant improvement for this role."

def get_match_level(score):
    if score >= 85:
        return "Excellent Match", "good"
    elif score >= 75:
        return "High Match", "good"
    elif score >= 60:
        return "Moderate Match", "mid"
    else:
        return "Low Match", "bad"

def render_tags(items, tag_type="found"):
    if not items:
        return "<span class='small-note'>No items detected.</span>"
    class_name = "tag-found" if tag_type == "found" else "tag-missing"
    html = ""
    for item in items:
        html += f"<span class='tag {class_name}'>{item}</span>"
    return html

def read_uploaded_resume(uploaded_file):
    if uploaded_file is None:
        return "", "No file uploaded."

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".txt"):
        try:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            if text.strip():
                return text, "TXT file loaded successfully."
            return "", "TXT file is empty."
        except Exception as e:
            return "", f"Could not read TXT file: {str(e)}"

    if file_name.endswith(".pdf"):
        pdf_bytes = uploaded_file.read()

        if PDF_PLUMBER_AVAILABLE:
            try:
                import io
                with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            pages_text.append(extracted)
                    text = "\n".join(pages_text).strip()
                    if text:
                        return text, "PDF loaded successfully with pdfplumber."
            except Exception:
                pass

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                pages_text = []
                for page in doc:
                    extracted = page.get_text("text")
                    if extracted:
                        pages_text.append(extracted)
                text = "\n".join(pages_text).strip()
                if text:
                    return text, "PDF loaded successfully with PyMuPDF."
            except Exception:
                pass

        if PYPDF2_AVAILABLE:
            try:
                import io
                reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                pages_text = []
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        pages_text.append(extracted)
                text = "\n".join(pages_text).strip()
                if text:
                    return text, "PDF loaded successfully with PyPDF2."
            except Exception:
                pass

        return "", (
            "Could not extract text from the uploaded PDF. "
            "Please use a text-based PDF (not a scanned image), or upload a TXT file."
        )

    return "", "Unsupported file format. Please upload PDF or TXT."

def create_score_chart(score_percent):
    remaining = max(0.0, 100.0 - float(score_percent))

    if score_percent >= 85:
        color = "#22c55e"
    elif score_percent >= 60:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.pie(
        [score_percent, remaining],
        startangle=90,
        counterclock=False,
        colors=[color, "#2d3748"],
        wedgeprops={"width": 0.28, "edgecolor": "none"}
    )

    ax.text(0, 0.08, "Match Score", ha="center", va="center", fontsize=14, color="white")
    ax.text(0, -0.12, f"{score_percent:.1f}%", ha="center", va="center", fontsize=28, fontweight="bold", color="white")
    ax.set(aspect="equal")
    return fig

def build_report_text(result, score, job_role, job_description, required_skills_text):
    lines = []
    lines.append("AI RESUME ANALYZER REPORT")
    lines.append("=" * 50)
    lines.append(f"Job Role: {job_role if job_role else 'Not Specified'}")
    lines.append(f"Final Match Score: {score}%")
    lines.append(f"Semantic Score: {result['semantic_score']}%")
    lines.append(f"Skill Score: {result['skill_score']}%")
    lines.append(f"Recommendation: {result['recommendation']}")
    lines.append("")
    lines.append("Job Description:")
    lines.append(job_description if job_description else "Not provided")
    lines.append("")
    lines.append("Required Skills:")
    lines.append(required_skills_text if required_skills_text else "Not provided")
    lines.append("")
    lines.append("Matched Required Skills:")
    if result["matched_required_skills"]:
        for s in result["matched_required_skills"]:
            lines.append(f"- {s}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Missing Skills:")
    if result["missing_skills"]:
        for s in result["missing_skills"]:
            lines.append(f"- {s}")
    else:
        lines.append("- None")
    return "\n".join(lines)

# =========================
# Load artifacts
# =========================
@st.cache_resource
def load_model_assets():
    with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    embedding_model = SentenceTransformer(config["embedding_model_name"])
    job_catalog = pd.read_csv(JOB_CATALOG_PATH)
    job_embeddings = np.load(JOB_EMBEDDINGS_PATH)

    with open(SKILL_LEXICON_PATH, "r", encoding="utf-8") as f:
        skill_lexicon = json.load(f)

    return embedding_model, job_catalog, job_embeddings, skill_lexicon, config

embedding_model, job_catalog, job_embeddings, skill_lexicon, model_config = load_model_assets()

# =========================
# ML logic
# =========================
def compute_match(resume_text, job_role, job_description, required_skills_text):
    resume_text_clean = clean_text(resume_text)
    job_role_clean = clean_text(job_role)
    job_description_clean = clean_text(job_description)

    resume_embed_text = f"resume: {resume_text_clean}"
    job_embed_text = f"role: {job_role_clean} [SEP] description: {job_description_clean} [SEP] skills: {required_skills_text}"

    resume_emb = embedding_model.encode([resume_embed_text], normalize_embeddings=True)
    job_emb = embedding_model.encode([job_embed_text], normalize_embeddings=True)

    cos_sim = float(cosine_similarity(resume_emb, job_emb)[0][0])
    semantic_score = round(semantic_score_from_cosine(cos_sim), 2)

    required_skills = split_skills(required_skills_text)
    resume_skills = extract_skills_from_text(resume_text, skill_lexicon)

    matched_required = []
    missing_skills = []

    cleaned_resume = clean_text(resume_text)

    for skill in required_skills:
        skill_norm = normalize_skill_token(skill)
        if skill_norm in resume_skills or skill_norm in cleaned_resume:
            matched_required.append(skill_norm)
        else:
            missing_skills.append(skill_norm)

    if len(required_skills) > 0:
        skill_score = round((len(matched_required) / len(required_skills)) * 100, 2)
    else:
        skill_score = 0.0

    final_score = hybrid_final_score(
        semantic_score=semantic_score,
        skill_score=skill_score,
        missing_count=len(missing_skills)
    )

    recommendation = build_recommendation(final_score, len(missing_skills))

    return {
        "semantic_score": semantic_score,
        "skill_score": skill_score,
        "final_score": final_score,
        "resume_skills_found": sorted(list(set(resume_skills))),
        "matched_required_skills": sorted(list(set(matched_required))),
        "missing_skills": sorted(list(set(missing_skills))),
        "recommendation": recommendation
    }

def recommend_top_jobs(resume_text, top_k=5):
    resume_text_clean = clean_text(resume_text)
    resume_embed_text = f"resume: {resume_text_clean}"
    resume_emb = embedding_model.encode([resume_embed_text], normalize_embeddings=True)

    results = []

    for idx, row in job_catalog.iterrows():
        job_role = row["job_role"]
        job_description = row["job_description"]
        required_skills_text = row["required_skills"]

        job_emb = job_embeddings[idx:idx+1]
        cos_sim = float(cosine_similarity(resume_emb, job_emb)[0][0])
        semantic_score = round(semantic_score_from_cosine(cos_sim), 2)

        required_skills = split_skills(required_skills_text)
        resume_skills = extract_skills_from_text(resume_text, skill_lexicon)

        matched_required = []
        missing_skills = []
        cleaned_resume = clean_text(resume_text)

        for skill in required_skills:
            skill_norm = normalize_skill_token(skill)
            if skill_norm in resume_skills or skill_norm in cleaned_resume:
                matched_required.append(skill_norm)
            else:
                missing_skills.append(skill_norm)

        if len(required_skills) > 0:
            skill_score = round((len(matched_required) / len(required_skills)) * 100, 2)
        else:
            skill_score = 0.0

        final_score = hybrid_final_score(
            semantic_score=semantic_score,
            skill_score=skill_score,
            missing_count=len(missing_skills)
        )

        results.append({
            "Job Role": str(job_role).title(),
            "Semantic Score (%)": semantic_score,
            "Skill Score (%)": skill_score,
            "Final Score (%)": final_score,
            "Missing Skills Count": len(missing_skills),
            "Missing Skills": ", ".join(missing_skills[:10]) if missing_skills else "None"
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["Final Score (%)", "Skill Score (%)", "Semantic Score (%)"],
        ascending=[False, False, False]
    ).head(top_k).reset_index(drop=True)

    return results_df

# =========================
# Session state
# =========================
defaults = {
    "resume_text": "",
    "job_role": "",
    "job_description": "",
    "required_skills_text": "",
    "uploaded_resume_name": "",
    "analysis_ready": False,
    "analysis_result": None,
    "analysis_score": None,
    "recommendations_ready": False,
    "recommendations_df": None,
    "recommendations_best_role": "",
    "recommendations_best_score": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def load_demo_data():
    st.session_state.resume_text = """Michael Johnson
Data Analyst

Summary:
Experienced Data Analyst with strong expertise in Python, SQL, Power BI, Excel, Tableau, Statistics, Data Analysis, Machine Learning, Pandas, and NumPy. Skilled in analyzing large datasets, building dashboards, automating reporting, and delivering business insights.

Skills:
Python, SQL, Power BI, Excel, Tableau, Statistics, Data Analysis, Machine Learning, Pandas, NumPy, KPI Reporting, Dashboard Development, Business Intelligence, ETL, Data Visualization

Experience:
Senior Data Analyst — Insight Analytics
- Built interactive dashboards in Power BI and Tableau
- Performed data analysis using Python, SQL, Pandas, and NumPy
- Automated reporting in Excel and Python
- Applied Statistics and Machine Learning for forecasting
- Delivered actionable business insights through KPI analysis

Education:
Bachelor’s Degree in Data Science
"""
    st.session_state.job_role = "Data Analyst"
    st.session_state.job_description = """We are looking for a Data Analyst with expert-level experience in Python, SQL, Power BI, Excel, Tableau, Statistics, Data Analysis, Machine Learning, Pandas, NumPy, KPI Reporting, Dashboard Development, Business Intelligence, ETL, and Data Visualization. The candidate must analyze large datasets, build dashboards, automate reporting, and provide actionable business insights."""
    st.session_state.required_skills_text = "Python, SQL, Power BI, Excel, Tableau, Statistics, Data Analysis, Machine Learning, Pandas, NumPy, KPI Reporting, Dashboard Development, Business Intelligence, ETL, Data Visualization"

def clear_analysis():
    st.session_state.analysis_ready = False
    st.session_state.analysis_result = None
    st.session_state.analysis_score = None

def clear_recommendations():
    st.session_state.recommendations_ready = False
    st.session_state.recommendations_df = None
    st.session_state.recommendations_best_role = ""
    st.session_state.recommendations_best_score = None

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## Project Info")
    st.write("This app compares a candidate resume with a job description, predicts a hybrid match score, detects missing skills, and recommends the best matching roles.")

    if st.button("Load Demo Example", use_container_width=True):
        load_demo_data()
        clear_analysis()
        clear_recommendations()

    st.markdown("---")
    st.markdown("## Resume Upload")
    uploaded_file = st.file_uploader("Upload PDF or TXT resume", type=["pdf", "txt"])

    if uploaded_file is not None:
        extracted_text, upload_message = read_uploaded_resume(uploaded_file)
        if extracted_text.strip():
            st.session_state.resume_text = extracted_text
            st.session_state.uploaded_resume_name = uploaded_file.name
            clear_analysis()
            clear_recommendations()
            st.success(f"Loaded: {uploaded_file.name}")
            st.caption(upload_message)
        else:
            st.warning(upload_message)

    if st.session_state.uploaded_resume_name:
        st.markdown("**Current uploaded file:**")
        st.caption(st.session_state.uploaded_resume_name)

    if st.session_state.resume_text.strip():
        with st.expander("Resume Preview"):
            preview = st.session_state.resume_text[:1500]
            st.markdown(f"<div class='preview-box'>{preview}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## Actions")
    if st.button("Clear Analysis Result", use_container_width=True):
        clear_analysis()

    if st.button("Clear Recommendations", use_container_width=True):
        clear_recommendations()

    st.markdown("---")
    st.markdown("## Model")
    st.write("Hybrid Deep Learning Model")
    st.write("SentenceTransformer + Skill Coverage")
    st.write("Outputs: semantic score, skill score, final score")

# =========================
# Header
# =========================
st.markdown('<div class="main-title">AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Resume–Job Matching and Missing Skills Detection Using Neural Networks</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Resume vs Job Analysis", "Top Job Recommendations"])

# =========================
# Tab 1
# =========================
with tab1:
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Candidate Resume")
        resume_text = st.text_area(
            "Resume",
            value=st.session_state.resume_text,
            height=320,
            label_visibility="collapsed",
            placeholder="Paste resume text here..."
        )

    with col_right:
        st.markdown("### Job Details")
        job_role = st.text_input(
            "Job Role",
            value=st.session_state.job_role,
            placeholder="e.g. Data Analyst"
        )
        job_description = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=170,
            placeholder="Paste job description here..."
        )
        required_skills_text = st.text_area(
            "Required Skills",
            value=st.session_state.required_skills_text,
            height=120,
            placeholder="Python, SQL, Power BI, Excel..."
        )

    st.session_state.resume_text = resume_text
    st.session_state.job_role = job_role
    st.session_state.job_description = job_description
    st.session_state.required_skills_text = required_skills_text

    if st.button("Analyze Resume", use_container_width=True):
        if not resume_text.strip() or not job_description.strip():
            st.warning("Please provide both resume text and job description.")
        else:
            result = compute_match(
                resume_text=resume_text,
                job_role=job_role,
                job_description=job_description,
                required_skills_text=required_skills_text
            )
            st.session_state.analysis_result = result
            st.session_state.analysis_score = result["final_score"]
            st.session_state.analysis_ready = True

    if st.session_state.analysis_ready and st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        score = st.session_state.analysis_score
        level_text, level_type = get_match_level(score)

        st.markdown("## Analysis Result")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Final Match Score</div>
                <div class="metric-value">{score}%</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Job Role</div>
                <div class="metric-value" style="font-size:22px;">{st.session_state.job_role if st.session_state.job_role else "Not Specified"}</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Missing Skills</div>
                <div class="metric-value">{len(result["missing_skills"])}</div>
            </div>
            """, unsafe_allow_html=True)

        st.progress(min(max(score / 100.0, 0.0), 1.0))

        if level_type == "good":
            st.markdown(f'<div class="good-box">{level_text}</div>', unsafe_allow_html=True)
        elif level_type == "mid":
            st.markdown(f'<div class="mid-box">{level_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bad-box">{level_text}</div>', unsafe_allow_html=True)

        extra1, extra2 = st.columns(2)
        with extra1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Semantic Score</div>
                <div class="metric-value">{result["semantic_score"]}%</div>
            </div>
            """, unsafe_allow_html=True)
        with extra2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Skill Coverage</div>
                <div class="metric-value">{result["skill_score"]}%</div>
            </div>
            """, unsafe_allow_html=True)

        chart_col, rec_col = st.columns([1, 1.4])

        with chart_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Score Chart")
            fig = create_score_chart(score)
            st.pyplot(fig, clear_figure=True, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rec_col:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Recommendation")
            st.write(result["recommendation"])
            st.write(f"Candidate fit for **{st.session_state.job_role if st.session_state.job_role else 'this role'}** is estimated at **{score}%**.")
            st.write(f"Semantic similarity score: **{result['semantic_score']}%**.")
            st.write(f"Skill coverage score: **{result['skill_score']}%**.")
            st.write(f"Identified **{len(result['missing_skills'])}** missing required skills.")
            st.markdown('</div>', unsafe_allow_html=True)

        left, right = st.columns(2)

        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Matched Required Skills")
            st.markdown(render_tags(result["matched_required_skills"], "found"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Resume Skills Found")
            st.markdown(render_tags(result["resume_skills_found"], "found"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("### Missing Skills")
            st.markdown(render_tags(result["missing_skills"], "missing"), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        report_text = build_report_text(
            result=result,
            score=score,
            job_role=st.session_state.job_role,
            job_description=st.session_state.job_description,
            required_skills_text=st.session_state.required_skills_text
        )

        st.download_button(
            label="Download Analysis Report",
            data=report_text,
            file_name="resume_analysis_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# =========================
# Tab 2
# =========================
with tab2:
    st.markdown("### Get Top Matching Job Roles")

    rec_resume_text = st.text_area(
        "Resume Text for Recommendations",
        value=st.session_state.resume_text,
        height=300,
        placeholder="Paste resume text here..."
    )

    top_k = st.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

    if st.button("Recommend Jobs", use_container_width=True):
        if not rec_resume_text.strip():
            st.warning("Please provide resume text.")
        else:
            recommendations = recommend_top_jobs(rec_resume_text, top_k=top_k)
            st.session_state.recommendations_df = recommendations
            st.session_state.recommendations_ready = True
            st.session_state.recommendations_best_role = recommendations.iloc[0]["Job Role"]
            st.session_state.recommendations_best_score = recommendations.iloc[0]["Final Score (%)"]

    if st.session_state.recommendations_ready and st.session_state.recommendations_df is not None:
        st.dataframe(
            st.session_state.recommendations_df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown(f"""
        <div class="section-card">
            <h4 style="margin-top:0;">Best Recommendation</h4>
            <p style="font-size:18px; margin-bottom:6px;"><strong>{st.session_state.recommendations_best_role}</strong></p>
            <p style="margin:0;">Estimated final match score: <strong>{st.session_state.recommendations_best_score}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.markdown(
    '<div class="footer-note">Developed for Neural Networks Project: Resume–Job Matching + Missing Skills Detection</div>',
    unsafe_allow_html=True
)
