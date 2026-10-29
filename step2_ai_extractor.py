"""
Step 2: AI Extraction Engine (Structured Outputs)
--------------------------------------------------
Takes cleaned resume text, sends it to a local LLM via Ollama,
and forces strict JSON output matching a Pydantic schema.
"""

import json
import re
import pymupdf as fitz
from ollama import chat
from pydantic import BaseModel, Field
from typing import List

# --- CONFIGURATION ---
PDF_PATH = "sanjeet-1"
MODEL = "llama3.2"  # Faster & smaller than 3.1, great for structured extraction


# ── Step 1 helpers (inline so this script is standalone) ────────────────────

def extract_text_from_pdf(path: str) -> str:
    raw_pages = []
    with fitz.open("Resume - ME.pdf") as doc:
        for page in doc:
            raw_pages.append(page.get_text("text"))
    return "\n".join(raw_pages)


def clean_text(raw: str) -> str:
    text = re.sub(r'[\u200b\u200c\u200d\u00ad\ufeff\u2060]', '', raw)
    text = re.sub(r'[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)
    text = re.sub(r'[^\x09\x0a\x0d\x20-\x7e\u00c0-\u024f]', '', text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r'([a-z,;])\n([a-z])', r'\1 \2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Pydantic Schema ───────────────────────────────────────────────────────────

class ResumeData(BaseModel):
    full_name: str = Field(description="Full name of the candidate")
    email: str = Field(description="Email address")
    skills: List[str] = Field(description="List of technical and soft skills")
    years_of_experience: float = Field(description="Total years of professional experience")
    education: List[str] = Field(description="Degrees or certifications obtained")
    job_titles: List[str] = Field(description="Previous job titles held")


# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_prompt(resume_text: str) -> str:
    example = json.dumps({
        "full_name": "Jane Doe",
        "email": "jane@example.com",
        "skills": ["Python", "React", "SQL"],
        "years_of_experience": 3.5,
        "education": ["B.Tech Computer Science, XYZ University (2021)"],
        "job_titles": ["Software Engineer", "Intern"]
    }, indent=2)

    return f"""You are a resume data extraction engine. Read the resume and extract information.

OUTPUT RULES (strictly follow):
- Output ONLY a single raw JSON object.
- No markdown, no explanation, no code fences, no extra text before or after.
- Use exactly these keys: full_name, email, skills, years_of_experience, education, job_titles
- skills must be a JSON array of strings
- education must be a JSON array of strings
- job_titles must be a JSON array of strings
- years_of_experience must be a number (e.g. 2.0)
- If a field is not found, use an empty string "" or empty array []

EXAMPLE OUTPUT FORMAT (use this exact structure, fill with real data from the resume):
{example}

RESUME TEXT:
\"\"\"
{resume_text}
\"\"\"

JSON OUTPUT:"""


# ── LLM Extraction ────────────────────────────────────────────────────────────

def extract_resume_data(resume_text: str) -> ResumeData:
    prompt = build_prompt(resume_text)

    print(f"[*] Sending to Ollama ({MODEL})...\n")

    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},  # Deterministic output
    )

    raw_output = response.message.content.strip()

    # Safety net: strip accidental markdown code fences if model adds them
    raw_output = re.sub(r'^```(?:json)?\s*', '', raw_output, flags=re.MULTILINE)
    raw_output = re.sub(r'```\s*$', '', raw_output, flags=re.MULTILINE)
    raw_output = raw_output.strip()

    # Debug: print raw LLM output before validation
    print("[DEBUG] Raw LLM output:")
    print(raw_output[:500])  # first 500 chars
    print("---")

    # Parse and validate against Pydantic schema
    try:
        parsed = ResumeData.model_validate_json(raw_output)
    except Exception as e:
        print(f"[ERROR] Pydantic validation failed: {e}")
        print("[HINT] Full raw output:")
        print(raw_output)
        raise
    return parsed


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Extract & clean
    print(f"[*] Reading PDF: {PDF_PATH}")
    raw = extract_text_from_pdf(PDF_PATH)
    clean = clean_text(raw)
    print(f"[✓] Text extracted ({len(clean)} chars)\n")

    # Step 2: LLM extraction
    resume_data = extract_resume_data(clean)

    print("=" * 60)
    print("EXTRACTED RESUME DATA (Validated JSON):")
    print("=" * 60)
    print(resume_data.model_dump_json(indent=2))
    print("=" * 60)
    print(f"\n[✓] Done. Found {len(resume_data.skills)} skills.")
