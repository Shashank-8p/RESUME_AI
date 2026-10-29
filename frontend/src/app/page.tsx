"use client";

import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";

// ── Types ──────────────────────────────────────────────────────────────────
interface RequirementRow {
  requirement: string;
  best_match: string | null;
  similarity: number;
  covered: boolean;
}

interface AnalysisResult {
  candidate_name: string;
  candidate_email: string;
  years_of_experience: number;
  skills_found: string[];
  final_score: number;
  covered_count: number;
  total_requirements: number;
  missing_skills: string[];
  requirement_breakdown: RequirementRow[];
}

// ── Score Ring ─────────────────────────────────────────────────────────────
function ScoreRing({ score }: { score: number }) {
  const color =
    score >= 75 ? "text-emerald-400" :
    score >= 50 ? "text-yellow-400"  : "text-red-400";
  return (
    <div className={`text-center ${color}`}>
      <div className="text-7xl font-bold">{score}</div>
      <div className="text-xl text-gray-400 mt-1">/ 100</div>
    </div>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [status,  setStatus]  = useState<"idle" | "loading" | "done" | "error">("idle");
  const [result,  setResult]  = useState<AnalysisResult | null>(null);
  const [errMsg,  setErrMsg]  = useState("");

  const onDrop = useCallback(async (accepted: File[]) => {
    const file = accepted[0];
    if (!file) return;

    setStatus("loading");
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Unknown error");
      }
      const data: AnalysisResult = await res.json();
      setResult(data);
      setStatus("done");
    } catch (e: unknown) {
      setErrMsg(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "application/pdf": [".pdf"] },
    multiple: false,
  });

  return (
    <main className="max-w-4xl mx-auto px-4 py-12">
      {/* Header */}
      <h1 className="text-4xl font-bold text-center mb-2">
        Skill Gap Analyzer
      </h1>
      <p className="text-center text-gray-400 mb-10">
        Drop your resume PDF — get an instant AI-powered match score.
      </p>

      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors
          ${isDragActive
            ? "border-indigo-400 bg-indigo-950"
            : "border-gray-600 hover:border-indigo-500 bg-gray-900"
          }`}
      >
        <input {...getInputProps()} />
        <p className="text-lg text-gray-300">
          {isDragActive
            ? "Drop it here..."
            : "Drag & drop your resume PDF here, or click to browse"}
        </p>
        <p className="text-sm text-gray-500 mt-2">Only PDF files accepted</p>
      </div>

      {/* Loading */}
      {status === "loading" && (
        <div className="mt-8 text-center text-indigo-400 animate-pulse text-lg">
          Analyzing your resume... (this takes ~30–60s)
        </div>
      )}

      {/* Error */}
      {status === "error" && (
        <div className="mt-8 bg-red-950 border border-red-700 rounded-xl p-6 text-red-300">
          <strong>Error:</strong> {errMsg}
        </div>
      )}

      {/* Results */}
      {status === "done" && result && (
        <div className="mt-10 space-y-6">

          {/* Score Card */}
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-8 flex flex-col items-center gap-4">
            <ScoreRing score={result.final_score} />
            <p className="text-gray-400">
              {result.covered_count} of {result.total_requirements} job requirements matched
            </p>
          </div>

          {/* Candidate Info */}
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 grid grid-cols-2 gap-4 text-sm">
            <div><span className="text-gray-500">Name</span><p className="font-medium">{result.candidate_name || "—"}</p></div>
            <div><span className="text-gray-500">Email</span><p className="font-medium">{result.candidate_email || "—"}</p></div>
            <div><span className="text-gray-500">Experience</span><p className="font-medium">{result.years_of_experience} years</p></div>
            <div><span className="text-gray-500">Skills Found</span><p className="font-medium">{result.skills_found.length}</p></div>
          </div>

          {/* Missing Skills */}
          {result.missing_skills.length > 0 && (
            <div className="bg-gray-900 border border-red-800 rounded-xl p-6">
              <h2 className="font-semibold text-red-400 mb-3">
                Missing Skills ({result.missing_skills.length})
              </h2>
              <div className="flex flex-wrap gap-2">
                {result.missing_skills.map((s) => (
                  <span key={s} className="bg-red-950 text-red-300 text-xs px-3 py-1 rounded-full border border-red-800">
                    {s}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Breakdown Table */}
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6">
            <h2 className="font-semibold text-gray-200 mb-4">Requirement Breakdown</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-700">
                    <th className="text-left py-2 pr-4">Job Requirement</th>
                    <th className="text-left py-2 pr-4">Best Match</th>
                    <th className="text-right py-2 pr-4">Score</th>
                    <th className="text-right py-2">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {result.requirement_breakdown.map((row) => (
                    <tr key={row.requirement} className="border-b border-gray-800">
                      <td className="py-2 pr-4 text-gray-300">{row.requirement}</td>
                      <td className="py-2 pr-4 text-gray-400">{row.best_match ?? "—"}</td>
                      <td className="py-2 pr-4 text-right text-gray-400">{row.similarity}</td>
                      <td className="py-2 text-right">
                        {row.covered
                          ? <span className="text-emerald-400">✓ Covered</span>
                          : <span className="text-red-400">✗ Missing</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Skills Found */}
          <div className="bg-gray-900 border border-gray-700 rounded-xl p-6">
            <h2 className="font-semibold text-gray-200 mb-3">Skills Extracted from Resume</h2>
            <div className="flex flex-wrap gap-2">
              {result.skills_found.map((s) => (
                <span key={s} className="bg-indigo-950 text-indigo-300 text-xs px-3 py-1 rounded-full border border-indigo-800">
                  {s}
                </span>
              ))}
            </div>
          </div>

          {/* Re-analyze */}
          <button
            onClick={() => { setStatus("idle"); setResult(null); }}
            className="w-full py-3 rounded-xl border border-gray-600 text-gray-400 hover:border-indigo-500 hover:text-indigo-400 transition"
          >
            Analyze another resume
          </button>
        </div>
      )}
    </main>
  );
}

