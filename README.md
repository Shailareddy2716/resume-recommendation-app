# Candidate Recommendation Engine

## Overview
This Streamlit web app recommends the best matching candidates for a given job description based on semantic relevance. The app uses NLP techniques to embed both job descriptions and candidate resumes, then computes cosine similarity to rank the top candidates. Optionally, it provides an AI-generated summary explaining why each candidate is a good fit.

## Features
- Accepts a job description (text input)
- Extracts and parse resume text using file-type specific tools
- Computes similarity scores using TF-IDF and cosine similarity
- Displays most relevant candidates with their extracted names and scores
- Generated AI-powered summaries of candidate fit using Hugging Face’s BART model

## Approach & Assumptions
-Used TfidfVectorizer from scikit-learn for converting job descriptions and resume texts into TF-IDF vectors, enabling effective text similarity comparison.
-Cosine similarity (also from scikit-learn) is used to rank resumes by their semantic closeness to the job description.
-For resume parsing, docx2txt is used for .docx files and PyPDF2 for .pdf files, with in-memory file handling via Python's io module.
-AI-powered summaries are generated using the Hugging Face transformers pipeline with the "facebook/bart-large-cnn" model, providing context-aware explanations for candidate-job fit.
-The app assumes all documents are in English and that resume and job description content is relevant and comparable.
-No external database or persistent storage is used; all processing happens in-memory for privacy and speed.
-The workflow is fully automated within the Streamlit app, requiring no manual preprocessing or external tools.

## How to Run
1. Install requirements:

pip install -r requirements.txt

2. Launch app:

streamlit run app.py

## Files
- `app.py` — Main Streamlit app
- `requirements.txt` — All dependencies

## Notes
- For public sharing, deployed on Streamlit Cloud or use `ngrok` for a temporary public link from your local machine.
- Code is well-commented for clarity and maintainability.
- No data is stored: All processing is performed in-memory, ensuring privacy and temporary handling of all uploaded files.

---

**Contact:** Shaila Reddy Kankanala
**Assignment for:** SproutsAI
