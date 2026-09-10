import React from "react";
import { ShieldCheck, AlertTriangle, XCircle, HelpCircle, Image as ImageIcon, Cpu, FileJson, Lock } from "lucide-react";
import CertificateViewer from "./CertificateViewer";

export default function ImageSentinelCard({ imageResult }) {
  if (!imageResult) return null;

  const { status, risk_score, evidence_strength, safety_result, transformation_metrics, findings, limitations_disclaimer } = imageResult;

  const isResourceLimitRejection = findings?.some(f => f.finding_type === "REJECTED_RESOURCE_LIMIT");

  const getStatusBadge = () => {
    if (status === "REJECTED" && isResourceLimitRejection) {
      return {
        bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
        icon: <AlertTriangle className="w-6 h-6" />,
        label: "REJECTED (RESOURCE BUDGET EXCEEDED)",
        subtext: "Image exceeded active resource policy profile (processing safety budget refusal, not proof of malicious intent).",
      };
    }
    switch (status) {
      case "SAFE_UNDER_CHECKS":
        return {
          bg: "bg-emerald-500/10 border-emerald-500/30 text-emerald-400",
          icon: <ShieldCheck className="w-6 h-6" />,
          label: "SAFE UNDER CHECKS",
          subtext: "Passed integrity, decoding, transformation-stability & reference baseline checks.",
        };
      case "SUSPICIOUS":
        return {
          bg: "bg-amber-500/10 border-amber-500/30 text-amber-400",
          icon: <AlertTriangle className="w-6 h-6" />,
          label: "SUSPICIOUS INPUT",
          subtext: "Exhibits prediction instability, distribution shift, or anomalous indicators.",
        };
      case "REJECTED":
        return {
          bg: "bg-rose-500/10 border-rose-500/30 text-rose-400",
          icon: <XCircle className="w-6 h-6" />,
          label: "REJECTED INPUT",
          subtext: "Failed fail-closed format validation, magic byte checks, or security rules.",
        };
      default:
        return {
          bg: "bg-slate-500/10 border-slate-500/30 text-slate-300",
          icon: <HelpCircle className="w-6 h-6" />,
          label: "UNGUARANTEED",
          subtext: "Valid file structure, but reference baseline or model calibration is unavailable.",
        };
    }
  };

  const badge = getStatusBadge();

  return (
    <div className="space-y-6">
      {/* Main Verdict Panel */}
      <div className={`border rounded-2xl p-6 shadow-2xl backdrop-blur-md ${badge.bg}`}>
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div className="flex items-center space-x-4">
            <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
              {badge.icon}
            </div>
            <div>
              <div className="text-[10px] uppercase font-mono tracking-wider opacity-75">
                RONOVA Image Sentinel Assurance Status
              </div>
              <h2 className="text-xl font-extrabold tracking-tight font-mono">{badge.label}</h2>
              <p className="text-xs opacity-90 mt-0.5 font-sans">{badge.subtext}</p>
            </div>
          </div>

          <div className="flex items-center space-x-4 font-mono text-right">
            <div>
              <div className="text-[10px] opacity-75 uppercase">Risk Score</div>
              <div className="text-lg font-extrabold">{risk_score}/100</div>
            </div>
            <div>
              <div className="text-[10px] opacity-75 uppercase">Evidence</div>
              <div className="text-xs font-bold px-2.5 py-1 bg-slate-950/80 border border-slate-800 rounded-lg">
                {evidence_strength}
              </div>
            </div>
          </div>
        </div>

        {/* Rejection Reasons if REJECTED */}
        {status === "REJECTED" && safety_result?.rejection_reasons?.length > 0 && (
          <div className="mt-4 p-4 bg-slate-950/90 border border-slate-800 rounded-xl space-y-1.5 text-xs">
            <div className={`font-bold font-mono uppercase text-[10px] ${isResourceLimitRejection ? "text-amber-400" : "text-rose-400"}`}>
              {isResourceLimitRejection ? "Processing Safety Boundary Refusal:" : "Hard Rejection Reason(s):"}
            </div>
            {isResourceLimitRejection && (
              <p className="text-slate-400 text-[11px] font-sans pb-1 border-b border-slate-800/80">
                Notice: This rejection is a processing-safety resource control. It does NOT imply the image file is malicious or trojaned.
              </p>
            )}
            <ul className={`list-disc list-inside space-y-0.5 ${isResourceLimitRejection ? "text-amber-300/90" : "text-rose-300/90"}`}>
              {safety_result.rejection_reasons.map((reason, idx) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Safety & Decode Metrics Grid */}
      {safety_result && (
        <div className="grid grid-cols-2 gap-4 text-xs font-sans">
          {/* File Safety & Format Integrity Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="font-bold text-slate-200 font-mono flex items-center space-x-1.5">
                <ImageIcon className="w-4 h-4 text-cyan-400" />
                <span>Format & Isolated Decode Safety</span>
              </span>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Worker PID: {safety_result.decode_worker_metadata?.worker_process_id || "N/A"}
              </span>
            </div>

            <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
              <div className="flex justify-between">
                <span className="text-slate-500">Magic Bytes Format:</span>
                <span className="text-cyan-400 font-bold">{safety_result.original_format}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Canonical Re-encode:</span>
                <span className="text-emerald-400">{safety_result.canonical_format} (Stripped PNG)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Dimensions & Mode:</span>
                <span>{safety_result.dimensions[0]}x{safety_result.dimensions[1]} ({safety_result.color_mode})</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">EXIF Metadata Stripped:</span>
                <span className="text-emerald-400 font-bold">YES (Clean Pixels)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-500">Perceptual Hash (dHash):</span>
                <span className="text-cyan-400">{safety_result.perceptual_hash || "N/A"}</span>
              </div>
            </div>
          </div>

          {/* Transformation & Stability Metrics Card */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <span className="font-bold text-slate-200 font-mono flex items-center space-x-1.5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                <span>Transformation Stability Suite</span>
              </span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                6 Seeded Variants
              </span>
            </div>

            {transformation_metrics ? (
              <div className="space-y-1.5 font-mono text-[11px] text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-500">Top-Class Flip Rate:</span>
                  <span className={transformation_metrics.top_class_flip_rate > 0.25 ? "text-rose-400 font-bold" : "text-emerald-400"}>
                    {(transformation_metrics.top_class_flip_rate * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Avg JS Divergence:</span>
                  <span className={transformation_metrics.average_js_divergence > 0.15 ? "text-amber-400 font-bold" : "text-slate-200"}>
                    {transformation_metrics.average_js_divergence}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Baseline Top Confidence:</span>
                  <span className="text-cyan-400">{(transformation_metrics.baseline_confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Prediction Dist L2:</span>
                  <span>{transformation_metrics.prediction_distribution_distance ?? transformation_metrics.embedding_l2_distance}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Model Adapter:</span>
                  <span className="text-cyan-400 font-mono text-[10px]">{safety_result.adapter_name || "N/A"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Entropy Change:</span>
                  <span>{transformation_metrics.entropy_change}</span>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic text-[11px] py-4 text-center">
                Transformation suite analysis unavailable for rejected input.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mandatory Limitations Disclaimer Card */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 text-xs font-sans text-slate-400 space-y-1">
        <div className="flex items-center space-x-1.5 text-cyan-400 font-mono font-bold text-[11px]">
          <Lock className="w-3.5 h-3.5" />
          <span>Image Sentinel Scope & Limitations Disclaimer</span>
        </div>
        <p className="leading-relaxed text-[11px] text-slate-400">
          {limitations_disclaimer || "Passed configured integrity, decoding, transformation-stability, and reference-distribution checks. This does not guarantee the image is authentic, harmless, non-adversarial, or safe under all possible attack conditions."}
        </p>
      </div>

      {/* Certificate Viewer Component if valid certificate present */}
      {imageResult.canonical_certificate && (
        <CertificateViewer result={imageResult} />
      )}
    </div>
  );
}
