"""
Step 3: Skill Gap Scoring Algorithm (Deterministic Vector Math)
----------------------------------------------------------------
Takes candidate skills (from Step 2) and job requirements (mock data).
Uses sentence-transformers to embed both lists, then computes cosine
similarity to produce a deterministic match score and missing skills list.
"""

import numpy as np
# pyrefly: ignore[missing-import]
from sentence_transformers import SentenceTransformer

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
MODEL_NAME = "all-MiniLM-L6-v2"   # Fast, 80MB, runs fully on CPU

# Threshold: a candidate skill is considered to "cover" a job requirement
# if their cosine similarity is >= this value (0.0 – 1.0)
MATCH_THRESHOLD = 0.50

# ── HARDCODED TEST DATA ───────────────────────────────────────────────────────
# In production these come from Step 2 output
candidate_skills = [
    "Python",
    "Machine Learning",
    "Data Analysis",
    "SQL",
    "TensorFlow",
    "REST APIs",
    "Git",
]

# Mock job description requirements
job_requirements = [
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


# ── CORE ALGORITHM ────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Pure deterministic cosine similarity between two vectors."""
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def score_candidate(
    candidate_skills: list[str],
    job_requirements: list[str],
    model: SentenceTransformer,
    threshold: float = MATCH_THRESHOLD,
) -> dict:
    """
    For each job requirement, find the best-matching candidate skill
    via cosine similarity. A requirement is 'covered' if the best
    match score >= threshold.

    Returns a structured result dict with score and gap analysis.
    """
    print("[*] Embedding candidate skills...")
    candidate_embeddings = model.encode(candidate_skills, normalize_embeddings=True)

    print("[*] Embedding job requirements...")
    job_embeddings = model.encode(job_requirements, normalize_embeddings=True)

    covered = []
    missing = []
    requirement_scores = []

    for i, req in enumerate(job_requirements):
        req_vec = job_embeddings[i]

        # Find best matching candidate skill
        best_score = 0.0
        best_skill = None
        for j, skill in enumerate(candidate_skills):
            sim = cosine_similarity(candidate_embeddings[j], req_vec)
            if sim > best_score:
                best_score = sim
                best_skill = skill

        requirement_scores.append({
            "requirement": req,
            "best_match": best_skill,
            "similarity": round(best_score, 4),
            "covered": best_score >= threshold,
        })

        if best_score >= threshold:
            covered.append(req)
        else:
            missing.append(req)

    # Final score: % of job requirements covered (0–100, deterministic)
    final_score = round((len(covered) / len(job_requirements)) * 100, 1)

    return {
        "final_score": final_score,
        "covered_count": len(covered),
        "total_requirements": len(job_requirements),
        "missing_skills": missing,
        "requirement_breakdown": requirement_scores,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[*] Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[✓] Model loaded\n")

    result = score_candidate(candidate_skills, job_requirements, model)

    print("\n" + "=" * 60)
    print("SKILL MATCH RESULTS")
    print("=" * 60)
    print(f"\n  Final Match Score : {result['final_score']} / 100")
    print(f"  Requirements Met  : {result['covered_count']} / {result['total_requirements']}")

    print("\n  Per-Requirement Breakdown:")
    print(f"  {'Requirement':<35} {'Best Match':<25} {'Score':>6}  {'Status'}")
    print(f"  {'-'*35} {'-'*25} {'-'*6}  {'-'*6}")
    for r in result["requirement_breakdown"]:
        status = "COVERED" if r["covered"] else "MISSING"
        print(f"  {r['requirement']:<35} {str(r['best_match']):<25} {r['similarity']:>6}  {status}")

    print(f"\n  Missing Skills ({len(result['missing_skills'])}):")
    for skill in result["missing_skills"]:
        print(f"    - {skill}")

    print("=" * 60)
    print(f"\n[✓] Done.")

