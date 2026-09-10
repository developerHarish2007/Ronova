import React from "react";
import { ShieldCheck, ShieldAlert, AlertTriangle, CheckCircle2, Lock, Cpu } from "lucide-react";

export default function VerdictPanel({ policy, sha256, filename, manifestVerification }) {
  if (!policy) return null;

  const { verdict, risk_score, evidence_strength, hard_gate_fired, hard_gates_triggered, findings } = policy;

  const isAccept = verdict === "ACCEPT";
  const isReview = verdict === "REVIEW";
  const isQuarantine = verdict === "QUARANTINE";

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl backdrop-blur-xl space-y-6">
      {/* Top Banner */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-5">
        <div className="flex items-center space-x-4">
          <div className={`p-3.5 rounded-2xl border ${
            isAccept ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
            isReview ? "bg-amber-500/10 border-amber-500/30 text-amber-400" :
            "bg-rose-500/10 border-rose-500/30 text-rose-400"
          }`}>
            {isAccept && <ShieldCheck className="w-8 h-8" />}
            {isReview && <AlertTriangle className="w-8 h-8" />}
            {isQuarantine && <ShieldAlert className="w-8 h-8" />}
          </div>
          <div>
            <div className="text-[11px] font-mono tracking-wider text-slate-400 uppercase">
              Governed Assurance Verdict
            </div>
            <div className={`text-3xl font-black tracking-tight mt-0.5 ${
              isAccept ? "text-emerald-400" : isReview ? "text-amber-400" : "text-rose-400"
            }`}>
              {verdict}
            </div>
          </div>
        </div>

        {/* Risk & Strength Cards */}
        <div className="flex items-center space-x-4 font-mono text-xs">
          <div className="bg-slate-950/80 border border-slate-800 px-4 py-2.5 rounded-xl text-right">
            <div className="text-[10px] text-slate-400 uppercase tracking-wider">Overall Risk Score</div>
            <div className={`text-lg font-bold ${risk_score > 50 ? "text-rose-400" : "text-emerald-400"}`}>
              {risk_score} <span className="text-xs text-slate-500 font-normal">/ 100</span>
            </div>
          </div>
          <div className="bg-slate-950/80 border border-slate-800 px-4 py-2.5 rounded-xl text-right">
            <div className="text-[10px] text-slate-400 uppercase tracking-wider">Evidence Strength</div>
            <div className="text-lg font-bold text-cyan-400">{evidence_strength}</div>
          </div>
        </div>
      </div>

      {/* Artifact Metadata Bar */}
      <div className="grid grid-cols-3 gap-3 font-mono text-xs">
        <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">Target Artifact</div>
          <div className="font-semibold text-slate-200 truncate mt-0.5">{filename || "model.onnx"}</div>
        </div>
        <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">SHA-256 Digest</div>
          <div className="font-semibold text-cyan-400 truncate mt-0.5">{sha256 ? `${sha256.slice(0, 16)}...` : "—"}</div>
        </div>
        <div className="bg-slate-950/60 border border-slate-800/80 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">Approved Manifest Status</div>
          <div className={`font-semibold truncate mt-0.5 ${manifestVerification?.manifest_matched ? "text-emerald-400" : "text-rose-400"}`}>
            {manifestVerification?.manifest_matched ? "MATCHED (OK)" : "MISMATCH (UNAPPROVED)"}
          </div>
        </div>
      </div>

      {/* Hard Gate Trigger List */}
      {hard_gate_fired && hard_gates_triggered?.length > 0 && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 space-y-2">
          <div className="flex items-center space-x-2 text-xs font-semibold text-rose-400">
            <ShieldAlert className="w-4 h-4" />
            <span>Level 1 Hard Security Gates Fired ({hard_gates_triggered.length})</span>
          </div>
          <ul className="list-disc list-inside space-y-1 text-xs text-rose-300/90 font-mono">
            {hard_gates_triggered.map((gate, idx) => (
              <li key={idx}>
                <span className="font-bold">{gate}</span> &mdash; Instant QUARANTINE enforced (cannot be averaged away).
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Active Findings List */}
      <div className="space-y-3">
        <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center space-x-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>Active Pipeline Security Findings ({findings?.length || 0})</span>
        </h3>

        <div className="space-y-2.5">
          {findings?.map((finding, idx) => (
            <div key={idx} className="bg-slate-950/70 border border-slate-800/90 rounded-xl p-4 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className={`w-2 h-2 rounded-full ${finding.is_hard_gate ? "bg-rose-500" : "bg-cyan-400"}`} />
                  <span className="text-xs font-bold text-slate-200">{finding.title}</span>
                </div>
                <div className="flex items-center space-x-2 font-mono text-[10px]">
                  <span className="px-2 py-0.5 bg-slate-900 border border-slate-700 rounded text-slate-400">
                    Risk: {finding.risk_score}/100
                  </span>
                  <span className="px-2 py-0.5 bg-slate-900 border border-slate-700 rounded text-cyan-400">
                    Strength: {finding.evidence_strength}
                  </span>
                </div>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed font-sans">{finding.explanation}</p>

              {finding.limitations && (
                <div className="text-[11px] text-slate-400 bg-slate-900/50 p-2.5 rounded-lg border border-slate-800/50 font-sans italic">
                  <span className="font-semibold text-slate-400 not-italic">Limitations: </span>
                  {finding.limitations}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
