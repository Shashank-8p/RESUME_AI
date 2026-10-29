"""
Step 4: FastAPI Backend
-----------------------
Wraps Steps 1-3 into a single POST /analyze endpoint.
Accepts a PDF upload, returns score + skill gap as JSON.
"""

import io
import json
import re
import tempfile
import os

import numpy as np
import pymupdf as fitz
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import chat
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# ── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_MODEL     = "llama3.2"
EMBED_MODEL      = "all-MiniLM-L6-v2"
MATCH_THRESHOLD  = 0.50

# Mock job requirements — swap this out with real JD input in V2
JOB_REQUIREMENTS = [
    "Python programming",
    "Deep Learning",
    "Natural Language Processing",
    "SQL databases",
    "PyTorch",
    "Docker",
    "Cloud deployment (AWS or GCP)",
    "Data visualization",
    "REST API development",
    "Version control with Git",
]

# ── GLOBAL MODEL (loaded once on startup) ────────────────────────────────────
embedder: SentenceTransformer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    print("[*] Loading sentence-transformer model...")
    embedder = SentenceTransformer(EMBED_MODEL)
    print("[✓] Model ready.")
    yield

app = FastAPI(title="Skill Gap Analyzer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── PYDANTIC SCHEMAS ──────────────────────────────────────────────────────────

class ResumeData(BaseModel):
    full_name: str = Field(default="")
    email: str = Field(default="")
    skills: List[str] = Field(default=[])
    years_of_experience: float = Field(default=0.0)
    education: List[str] = Field(default=[])
    job_titles: List[str] = Field(default=[])


class AnalysisResult(BaseModel):
    candidate_name: str
    candidate_email: str
    years_of_experience: float
    skills_found: List[str]
    final_score: float
    covered_count: int
    total_requirements: int
    missing_skills: List[str]
    requirement_breakdown: List[dict]


# ── STEP 1: PDF EXTRACTION ────────────────────────────────────────────────────

def extract_and_clean(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        raw = "\n".join(page.get_text("text") for page in doc)

    text = re.sub(r'[\u200b\u200c\u200d\u00ad\ufeff\u2060]', '', raw)
    text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)
    text = re.sub(r'[^\x09\x0a\x0d\x20-\x7e\u00c0-\u024f]', '', text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── STEP 2: LLM EXTRACTION ───────────────────────────────────────────────────

def extract_resume_data(resume_text: str) -> ResumeData:
    example = json.dumps({
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "skills": ["Python", "React", "SQL"],
        "years_of_experience": 3.5,
        "education": ["B.Tech Computer Science, XYZ University (2021)"],
        "job_titles": ["Software Engineer", "Intern"]
    }, indent=2)

    prompt = f"""You are a resume data extraction engine. Read the resume and extract information.

OUTPUT RULES (strictly follow):
- Output ONLY a single raw JSON object.
- No markdown, no explanation, no code fences, no extra text.
- Use exactly these keys: full_name, email, skills, years_of_experience, education, job_titles
- skills, education, job_titles must be JSON arrays of strings
- years_of_experience must be a number
- If a field is not found, use "" or []

EXAMPLE OUTPUT FORMAT:
{example}

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

JSON OUTPUT:"""

    response = chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    raw = response.message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE).strip()

    return ResumeData.model_validate_json(raw)


# ── STEP 3: SCORING ───────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm else 0.0


def score_candidate(candidate_skills: List[str], job_requirements: List[str]) -> dict:
    cand_emb = embedder.encode(candidate_skills, normalize_embeddings=True)
    job_emb  = embedder.encode(job_requirements,  normalize_embeddings=True)

    covered, missing, breakdown = [], [], []

    for i, req in enumerate(job_requirements):
        best_score, best_skill = 0.0, None
        for j, skill in enumerate(candidate_skills):
            sim = cosine_similarity(cand_emb[j], job_emb[i])
            if sim > best_score:
                best_score, best_skill = sim, skill

        is_covered = best_score >= MATCH_THRESHOLD
        breakdown.append({
            "requirement": req,
            "best_match":  best_skill,
            "similarity":  round(best_score, 4),
            "covered":     is_covered,
        })
        (covered if is_covered else missing).append(req)

    return {
        "final_score":           round((len(covered) / len(job_requirements)) * 100, 1),
        "covered_count":         len(covered),
        "total_requirements":    len(job_requirements),
        "missing_skills":        missing,
        "requirement_breakdown": breakdown,
    }


# ── ENDPOINT ──────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    try:
        clean_text   = extract_and_clean(pdf_bytes)
        resume_data  = extract_resume_data(clean_text)
        score_result = score_candidate(resume_data.skills, JOB_REQUIREMENTS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AnalysisResult(
        candidate_name      = resume_data.full_name,
        candidate_email     = resume_data.email,
        years_of_experience = resume_data.years_of_experience,
        skills_found        = resume_data.skills,
        **score_result,
    )


@app.get("/")
def root():
    return {"status": "Skill Gap Analyzer API is running."}

