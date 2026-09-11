import React, { useState } from "react";
import { Database, AlertTriangle, CheckCircle, HelpCircle, Layers, Eye, Tag, BarChart2, ShieldAlert, ChevronLeft, ChevronRight, ChevronDown, ChevronUp } from "lucide-react";

const DUP_PAGE_SIZE = 10;

export default function DatasetForensicsCard({ datasetResult }) {
  const [selectedPoint, setSelectedPoint] = useState(null);
  const [dupPage, setDupPage] = useState(0);
  const [showDist, setShowDist] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(false);

  if (!datasetResult) return null;

  const {
    filename,
    dataset_role,
    analysis_protocol,
    array_shape = [],
    dtype,
    dimensions,
    channels,
    total_samples,
    dataset_risk_assessment,
    overall_risk_score = 0.0,
    duplicate_pairs_count = 0,
    duplicate_pairs = [],
    anomalous_samples_count = 0,
    anomalous_indices = [],
    anomaly_scores_summary = {},
    class_channel_distribution = {},
    chart_coordinates = [],
    flagged_thumbnails = [],
    findings = [],
    verdict_explanation,
    limitations_disclaimer,
  } = datasetResult;

  const getRoleBadge = (role) => {
    switch (role) {
      case "perturbation_overlay":
        return { label: "Perturbation Reference Overlay", bg: "bg-purple-500/10 border-purple-500/30 text-purple-400" };
      case "live_input_batch":
        return { label: "Live Inference Input Batch", bg: "bg-cyan-500/10 border-cyan-500/30 text-cyan-400" };
      default:
        return { label: "Training / Evaluation Dataset", bg: "bg-blue-500/10 border-blue-500/30 text-blue-400" };
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case "ACCEPT":
        return {
          label: "ACCEPT (Clean)",
          bg: "bg-emerald-500/10 border-emerald-500/40 text-emerald-400",
          icon: <CheckCircle className="w-5 h-5 text-emerald-400 shrink-0" />,
        };
      case "REVIEW":
        return {
          label: "REVIEW (Flagged)",
          bg: "bg-amber-500/10 border-amber-500/40 text-amber-400",
          icon: <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />,
        };
      default:
        return {
          label: "UNGUARANTEED",
          bg: "bg-purple-500/10 border-purple-500/40 text-purple-400",
          icon: <HelpCircle className="w-5 h-5 text-purple-400 shrink-0" />,
        };
    }
  };

  const roleInfo = getRoleBadge(dataset_role);
  const statusInfo = getStatusBadge(dataset_risk_assessment);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
      {/* Top Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-slate-800 pb-4 gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl text-cyan-400">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-bold text-white tracking-tight">{filename}</h2>
              <span className={`text-[10px] font-mono font-semibold px-2.5 py-0.5 rounded-full border ${roleInfo.bg}`}>
                {roleInfo.label}
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Protocol: {analysis_protocol || "CNN Embedding Isolation Forest"}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl border border-slate-800 bg-slate-950/80 font-mono text-xs">
            <span className="text-slate-500">Risk Score:</span>
            <span className={`font-bold ${overall_risk_score > 0 ? "text-amber-400" : "text-emerald-400"}`}>
              {overall_risk_score.toFixed(1)} / 100
            </span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-xl border border-slate-800 bg-slate-950/80 font-mono text-xs">
            <span className="text-slate-500">Evidence:</span>
            <span className="text-cyan-400 font-bold">{datasetResult.overall_evidence_strength || "HIGH"}</span>
          </div>

          <div className={`flex items-center space-x-2 px-4 py-2 rounded-xl border font-bold text-xs font-mono ${statusInfo.bg}`}>
            {statusInfo.icon}
            <span>{statusInfo.label}</span>
          </div>
        </div>
      </div>

      {/* Explanation Banner */}
      {verdict_explanation && (
        <div className="p-4 bg-slate-950/80 border border-slate-800 rounded-xl text-xs text-slate-300 font-sans leading-relaxed flex items-start space-x-3">
          <Layers className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-slate-100">Assessment Rationale: </span>
            {verdict_explanation}
          </div>
        </div>
      )}

      {/* Metric Cards Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono">
        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Total Samples</div>
          <div className="text-base font-bold text-white mt-1">{total_samples}</div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Array Shape</div>
          <div className="text-xs font-bold text-cyan-400 mt-1 truncate">
            {array_shape.length > 0 ? array_shape.join(" × ") : "N/A"}
          </div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Data Type</div>
          <div className="text-xs font-bold text-slate-300 mt-1">{dtype || "float32"}</div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Dimensions</div>
          <div className="text-xs font-bold text-slate-300 mt-1">{dimensions}D ({channels} ch)</div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Duplicate Pairs</div>
          <div className={`text-base font-bold mt-1 ${duplicate_pairs_count > 0 ? "text-amber-400" : "text-emerald-400"}`}>
            {duplicate_pairs_count}
            <span className="text-[10px] font-normal text-slate-400 ml-1.5">
              ({total_samples > 0 ? ((duplicate_pairs_count / total_samples) * 100).toFixed(1) : 0}%)
            </span>
          </div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Anomalies</div>
          <div className={`text-base font-bold mt-1 ${anomalous_samples_count > 0 ? "text-rose-400" : "text-emerald-400"}`}>
            {anomalous_samples_count}
            <span className="text-[10px] font-normal text-slate-400 ml-1.5">
              ({total_samples > 0 ? ((anomalous_samples_count / total_samples) * 100).toFixed(1) : 0}%)
            </span>
          </div>
        </div>
      </div>

      {/* 2D Embedding Anomaly Chart */}
      {chart_coordinates.length > 0 && (() => {
        const totalPts = chart_coordinates.length;
        const anomCount = chart_coordinates.filter(p => p.is_anomaly).length;
        const inlierCount = totalPts - anomCount;
        const anomRate = totalPts > 0 ? ((anomCount / totalPts) * 100).toFixed(1) : 0;
        const inlierRate = totalPts > 0 ? ((inlierCount / totalPts) * 100).toFixed(1) : 0;
        const threshold = anomaly_scores_summary?.threshold ?? -0.07;
        const minScore = anomaly_scores_summary?.min_score ?? -0.15;
        const maxScore = anomaly_scores_summary?.max_score ?? 0.15;

        const getAnomalyConfidence = (score) => {
          if (score === undefined || score === null) return "N/A";
          if (score < threshold) {
            // Anomaly region: scale distance past threshold to 70-99% confidence
            const delta = Math.abs(threshold - score);
            const span = Math.max(0.01, Math.abs(threshold - minScore));
            const conf = Math.min(99.4, 70.0 + (delta / span) * 29.4);
            return `${conf.toFixed(1)}% (High Risk)`;
          } else {
            // Normal region: scale proximity to maxScore as inlier confidence
            const delta = Math.abs(score - threshold);
            const span = Math.max(0.01, Math.abs(maxScore - threshold));
            const inlierConf = Math.min(99.8, 65.0 + (delta / span) * 34.8);
            return `${(100 - inlierConf).toFixed(1)}% (Clean Inlier)`;
          }
        };

        return (
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800/60 pb-3 gap-2">
              <div className="flex items-center space-x-2">
                <BarChart2 className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                  2D Feature Space Embedding & Anomaly Scatter Plot
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-3 text-[11px] font-mono">
                <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-cyan-500/10 border border-cyan-500/30">
                  <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block"></span>
                  <span className="text-cyan-300 font-semibold">{inlierRate}% Inliers ({inlierCount})</span>
                </div>
                <div className="flex items-center space-x-1.5 px-2 py-0.5 rounded bg-rose-500/10 border border-rose-500/30">
                  <span className="w-2 h-2 rounded-full bg-rose-500 inline-block animate-pulse"></span>
                  <span className="text-rose-300 font-semibold">{anomRate}% Anomalies ({anomCount})</span>
                </div>
              </div>
            </div>

            {/* Score Distribution Context Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono bg-slate-900/60 p-2.5 rounded-lg border border-slate-800/60">
              <div>
                <span className="text-slate-500">Isolation Threshold:</span>{" "}
                <span className="text-amber-400 font-bold">{threshold}</span>
              </div>
              <div>
                <span className="text-slate-500">Mean Score:</span>{" "}
                <span className="text-slate-300 font-bold">{anomaly_scores_summary?.mean_score ?? "0.0000"}</span>
              </div>
              <div>
                <span className="text-slate-500">Score Range:</span>{" "}
                <span className="text-slate-300">[{minScore}, {maxScore}]</span>
              </div>
              <div>
                <span className="text-slate-500">Dimensionality Reduction:</span>{" "}
                <span className="text-cyan-400">PCA (64D → 2D)</span>
              </div>
            </div>

            {/* Scatter SVG Area */}
            <div className="relative w-full h-72 bg-slate-900/40 rounded-lg overflow-hidden border border-slate-800/40 flex items-center justify-center p-4">
              <svg viewBox="-1.2 -1.2 2.4 2.4" className="w-full h-full">
                {/* Grid Lines */}
                <line x1="-1.2" y1="0" x2="1.2" y2="0" stroke="#334155" strokeWidth="0.005" strokeDasharray="0.03" />
                <line x1="0" y1="-1.2" x2="0" y2="1.2" stroke="#334155" strokeWidth="0.005" strokeDasharray="0.03" />
                
                {/* Active Hover Crosshairs */}
                {selectedPoint && (
                  <g className="transition-all duration-150">
                    <line x1={selectedPoint.x} y1="-1.2" x2={selectedPoint.x} y2="1.2" stroke={selectedPoint.is_anomaly ? "#f43f5e" : "#22d3ee"} strokeWidth="0.003" strokeDasharray="0.02" opacity="0.5" />
                    <line x1="-1.2" y1={-selectedPoint.y} x2="1.2" y2={-selectedPoint.y} stroke={selectedPoint.is_anomaly ? "#f43f5e" : "#22d3ee"} strokeWidth="0.003" strokeDasharray="0.02" opacity="0.5" />
                    <circle cx={selectedPoint.x} cy={-selectedPoint.y} r="0.08" fill="none" stroke={selectedPoint.is_anomaly ? "#f43f5e" : "#22d3ee"} strokeWidth="0.008" opacity="0.8" />
                  </g>
                )}

                {/* Normal Points */}
                {chart_coordinates.filter(p => !p.is_anomaly).map((pt, idx) => (
                  <g key={`norm-${idx}`} onMouseEnter={() => setSelectedPoint(pt)} onClick={() => setSelectedPoint(pt)} className="cursor-pointer">
                    {/* Invisible larger hit area for easy hover */}
                    <circle cx={pt.x} cy={-pt.y} r="0.05" fill="transparent" />
                    <circle
                      cx={pt.x}
                      cy={-pt.y}
                      r="0.025"
                      className="fill-cyan-400 opacity-70 hover:opacity-100 transition-all hover:scale-125"
                    />
                  </g>
                ))}

                {/* Anomalous Points */}
                {chart_coordinates.filter(p => p.is_anomaly).map((pt, idx) => (
                  <g key={`anom-${idx}`} onMouseEnter={() => setSelectedPoint(pt)} onClick={() => setSelectedPoint(pt)} className="cursor-pointer">
                    {/* Invisible larger hit area */}
                    <circle cx={pt.x} cy={-pt.y} r="0.07" fill="transparent" />
                    <circle cx={pt.x} cy={-pt.y} r="0.06" className="fill-rose-500/30 animate-ping" />
                    <circle cx={pt.x} cy={-pt.y} r="0.04" className="fill-rose-500 stroke-white stroke-1" />
                    <text x={pt.x + 0.05} y={-pt.y + 0.03} fontSize="0.06" fill="#f43f5e" fontFamily="monospace" fontWeight="bold">
                      #{pt.sample_idx}
                    </text>
                  </g>
                ))}
              </svg>

              {/* Hover / Selection Tooltip HUD */}
              <div className="absolute top-3 right-3 max-w-[240px] bg-slate-950/95 border border-slate-700/80 rounded-xl p-3 text-[11px] font-mono text-slate-200 shadow-2xl backdrop-blur-md">
                {selectedPoint ? (
                  <div className="space-y-1.5">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-1">
                      <span className="text-white font-bold">Sample #{selectedPoint.sample_idx}</span>
                      <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold ${selectedPoint.is_anomaly ? "bg-rose-500/20 text-rose-400 border border-rose-500/40" : "bg-cyan-500/20 text-cyan-400 border border-cyan-500/40"}`}>
                        {selectedPoint.is_anomaly ? "OUTLIER" : "INLIER"}
                      </span>
                    </div>
                    <div><span className="text-slate-400">Decision Score:</span> <span className={selectedPoint.score < threshold ? "text-rose-400 font-bold" : "text-cyan-300 font-bold"}>{selectedPoint.score}</span></div>
                    <div><span className="text-slate-400">Anomaly Prob:</span> <span className={selectedPoint.is_anomaly ? "text-rose-400 font-bold" : "text-emerald-400"}>{getAnomalyConfidence(selectedPoint.score)}</span></div>
                    <div><span className="text-slate-400">PCA Coords:</span> <span className="text-slate-300">({selectedPoint.x.toFixed(3)}, {selectedPoint.y.toFixed(3)})</span></div>
                  </div>
                ) : (
                  <div className="text-[10px] text-slate-400 italic flex items-center space-x-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
                    <span>Hover or click any point to inspect score, confidence & coords</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })()}

      {/* Flagged Anomaly Sample Thumbnails */}
      {flagged_thumbnails.length > 0 && (
        <div className="space-y-3 border-t border-slate-800 pt-4">
          <div className="flex items-center space-x-2 text-xs font-bold text-rose-400 font-mono">
            <Eye className="w-4 h-4" />
            <span>Flagged Anomalous Sample Visual Previews ({flagged_thumbnails.length})</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3 font-mono text-xs">
            {flagged_thumbnails.map((item, idx) => (
              <div key={idx} className="bg-slate-950 border border-rose-500/30 rounded-xl p-3 flex flex-col items-center space-y-2">
                {item.thumbnail_b64 ? (
                  <img
                    src={item.thumbnail_b64}
                    alt={`Sample ${item.sample_idx}`}
                    className="w-14 h-14 object-contain rounded-md border border-slate-800 bg-black"
                  />
                ) : (
                  <div className="w-14 h-14 bg-slate-900 border border-slate-800 rounded-md flex items-center justify-center text-[10px] text-slate-600">
                    No Image
                  </div>
                )}
                <div className="text-[11px] font-bold text-white">Sample #{item.sample_idx}</div>
                <div className="text-[10px] text-rose-400">Score: {item.anomaly_score}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Duplicate Pairs Section */}
      {duplicate_pairs.length > 0 ? (() => {
        const totalPages = Math.ceil(duplicate_pairs.length / DUP_PAGE_SIZE);
        const pageStart = dupPage * DUP_PAGE_SIZE;
        const pageEnd = Math.min(pageStart + DUP_PAGE_SIZE, duplicate_pairs.length);
        const pagePairs = duplicate_pairs.slice(pageStart, pageEnd);
        return (
          <div className="space-y-3 border-t border-slate-800 pt-4 font-mono text-xs">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 font-bold text-amber-400">
                <Tag className="w-4 h-4" />
                <span>Identified Duplicate &amp; Near-Duplicate Pairs ({duplicate_pairs.length})</span>
              </div>
              <span className="text-[11px] text-amber-400/80 bg-amber-500/10 border border-amber-500/30 px-2 py-0.5 rounded">
                Flagged for Data Redundancy / Poisoning
              </span>
            </div>

            <div className="overflow-x-auto border border-slate-800 rounded-xl">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-slate-950 text-slate-400 border-b border-slate-800 text-[10px] uppercase">
                    <th className="p-3">Sample 1</th>
                    <th className="p-3">Sample 2</th>
                    <th className="p-3">Hamming Distance</th>
                    <th className="p-3">MSE Distance</th>
                    <th className="p-3">Match Type</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 bg-slate-900/40">
                  {pagePairs.map((pair, idx) => (
                    <tr key={pageStart + idx} className="hover:bg-slate-800/40 transition">
                      <td className="p-3 font-semibold text-slate-200 flex items-center space-x-2">
                        {pair.thumbnail_1 && <img src={pair.thumbnail_1} className="w-6 h-6 rounded border border-slate-800 shrink-0" alt="" />}
                        <span>Sample #{pair.sample_idx_1}</span>
                      </td>
                      <td className="p-3 font-semibold text-slate-200 flex items-center space-x-2">
                        {pair.thumbnail_2 && <img src={pair.thumbnail_2} className="w-6 h-6 rounded border border-slate-800 shrink-0" alt="" />}
                        <span>Sample #{pair.sample_idx_2}</span>
                      </td>
                      <td className="p-3 text-slate-300">{pair.hamming_distance} bits</td>
                      <td className="p-3 text-slate-300">{pair.mse_distance}</td>
                      <td className="p-3">
                        {pair.is_exact_match ? (
                          <span className="px-2 py-0.5 text-[10px] bg-amber-500/10 border border-amber-500/30 text-amber-400 rounded-md font-bold">
                            EXACT MD5
                          </span>
                        ) : (
                          <span className="px-2 py-0.5 text-[10px] bg-cyan-500/10 border border-cyan-500/30 text-cyan-400 rounded-md font-semibold">
                            NEAR D-HASH
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination controls — only show if more than one page */}
            {totalPages > 1 && (
              <div className="flex items-center justify-between pt-1">
                <span className="text-[10px] text-slate-500">
                  Showing {pageStart + 1}–{pageEnd} of {duplicate_pairs.length}
                </span>
                <div className="flex items-center space-x-1">
                  <button
                    onClick={() => setDupPage((p) => Math.max(0, p - 1))}
                    disabled={dupPage === 0}
                    className="flex items-center space-x-1 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition"
                  >
                    <ChevronLeft className="w-3.5 h-3.5" />
                    <span>Prev</span>
                  </button>
                  <span className="text-[10px] text-slate-500 px-2">{dupPage + 1} / {totalPages}</span>
                  <button
                    onClick={() => setDupPage((p) => Math.min(totalPages - 1, p + 1))}
                    disabled={dupPage === totalPages - 1}
                    className="flex items-center space-x-1 px-2.5 py-1 rounded-lg border border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700 disabled:opacity-30 disabled:cursor-not-allowed transition"
                  >
                    <span>Next</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            )}
          </div>
        );
      })() : (
        <div className="border border-slate-800 rounded-xl p-4 bg-slate-950/60 font-mono text-xs space-y-2 border-t border-slate-800 pt-4">
          <div className="flex items-center space-x-2 text-emerald-400 font-bold">
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>Perceptual &amp; Exact Duplicate Forensic Scan — 100% Unique (0 Pairs)</span>
          </div>
          <p className="text-[11px] text-slate-400 font-sans leading-relaxed">
            All {total_samples} samples cross-evaluated against 64-bit dHash perceptual distance (Hamming threshold ≤ 4 bits) and MSE pixel difference (threshold &lt; 0.001). 0 duplicate or near-clone pairs detected.
          </p>
        </div>
      )}

      {/* Distribution Summary — collapsible */}
      {class_channel_distribution.mean_pixel !== undefined && (
        <div className="border border-slate-800 rounded-xl font-mono text-xs overflow-hidden">
          <button
            onClick={() => setShowDist((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 bg-slate-950/60 hover:bg-slate-800/40 transition text-left"
          >
            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">
              Dataset Feature Pixel Distribution Statistics
            </span>
            {showDist
              ? <ChevronUp className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              : <ChevronDown className="w-3.5 h-3.5 text-slate-500 shrink-0" />}
          </button>
          {showDist && (
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-slate-300 px-4 py-3 bg-slate-950/30 border-t border-slate-800">
              <div><span className="text-slate-500">Pixel Mean:</span> {class_channel_distribution.mean_pixel}</div>
              <div><span className="text-slate-500">Pixel Std:</span> {class_channel_distribution.std_pixel}</div>
              <div><span className="text-slate-500">Pixel Min:</span> {class_channel_distribution.min_pixel}</div>
              <div><span className="text-slate-500">Pixel Max:</span> {class_channel_distribution.max_pixel}</div>
            </div>
          )}
        </div>
      )}

      {/* Honest Limitations Disclaimer — collapsible */}
      {limitations_disclaimer && (
        <div className="border border-slate-800/80 rounded-xl text-xs overflow-hidden">
          <button
            onClick={() => setShowDisclaimer((v) => !v)}
            className="w-full flex items-center justify-between px-4 py-3 bg-slate-950 hover:bg-slate-800/40 transition text-left"
          >
            <div className="flex items-center space-x-1.5 font-semibold text-slate-300">
              <ShieldAlert className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>Honest Assurance &amp; Limitations Disclaimer</span>
            </div>
            {showDisclaimer
              ? <ChevronUp className="w-3.5 h-3.5 text-slate-500 shrink-0" />
              : <ChevronDown className="w-3.5 h-3.5 text-slate-500 shrink-0" />}
          </button>
          {showDisclaimer && (
            <p className="text-[11px] leading-relaxed text-slate-400 px-4 py-3 bg-slate-950/50 border-t border-slate-800/60">{limitations_disclaimer}</p>
          )}
        </div>
      )}
    </div>
  );
}
