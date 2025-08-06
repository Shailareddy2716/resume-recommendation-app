import streamlit as st
import docx2txt
import PyPDF2
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# Load Hugging Face summarizer (free, runs locally)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

def extract_text_from_pdf(file_buffer):
    pdf_reader = PyPDF2.PdfReader(file_buffer)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def extract_text_from_docx(file_buffer):
    return docx2txt.process(file_buffer)

def process_file(file_bytes, file_name):
    file_buffer = io.BytesIO(file_bytes)
    if file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_buffer)
    elif file_name.endswith('.docx'):
        return extract_text_from_docx(file_buffer)
    else:
        return ""

def extract_candidate_name(resume_text):
    for line in resume_text.splitlines():
        line = line.strip()
        if line and len(line.split()) <= 4 and line[0].isupper():
            return line
    return "Unknown"

def get_free_summary(job_desc, resume_text):
    prompt = (
        f"Given this job description: {job_desc}\n"
        f"And this candidate resume: {resume_text}\n"
        "Briefly summarize why or why not this candidate fits the job."
    )
    # Truncate input to fit summarizer limits
    input_text = prompt[:1024]
    summary = summarizer(input_text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

st.markdown(
    "<p style='font-size:36px; font-weight:700; text-align:center;'>Candidate Recommendation System</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<div style='font-size:15px; color: #555; text-align:center;'>"
    "Upload DOCX/PDF resumes or paste resume text below. For multiple pasted resumes, separate each with a line containing only three dashes (---).<br>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

job_desc = st.text_area("Paste Job Description")

st.markdown("<br>", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload Candidate Resumes (DOCX/PDF, multiple allowed)",
    type=['pdf', 'docx'],
    accept_multiple_files=True
)

resume_text_input = st.text_area(
    "Or paste one or more resumes below. For multiple resumes, separate each resume by a line with only three dashes (---):"
)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Find Best Match"):
    resumes_text = []
    if uploaded_files:
        for f in uploaded_files:
            f.seek(0)
            resumes_text.append(process_file(f.read(), f.name))
    if resume_text_input.strip():
        pasted_resumes = [r.strip() for r in resume_text_input.split('\n---\n') if r.strip()]
        resumes_text.extend(pasted_resumes)

    if resumes_text:
        documents = [job_desc] + resumes_text
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        sorted_indices = similarities.argsort()[::-1]
        num_resumes = len(resumes_text)
        top_n = min(10, num_resumes)

        if num_resumes < 10:
            st.success("Top candidates by similarity score:")
        else:
            st.success("Top 10 candidates by similarity score:")

        for rank, idx in enumerate(sorted_indices[:top_n], 1):
            candidate_name = extract_candidate_name(resumes_text[idx])
            score = similarities[idx]
            st.markdown(
               f"{rank}. <b>Name:</b> {candidate_name}<br>"
               f"<b>Similarity Score:</b> {score:.2f}<br><br>"
               f"<i><b>Fit Insights:</b> {get_free_summary(job_desc, resumes_text[idx])}</i>",
               unsafe_allow_html=True
            )
    else:
        st.warning("Please upload or paste at least one resume.")

st.markdown("<br>", unsafe_allow_html=True)
st.info("Tip: For best results, paste or upload resumes in standard formats with the name at the top.")
