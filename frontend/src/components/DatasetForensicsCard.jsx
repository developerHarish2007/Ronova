import React, { useState } from "react";
import { Database, AlertTriangle, CheckCircle, HelpCircle, Layers, Eye, Tag, BarChart2, ShieldAlert } from "lucide-react";

export default function DatasetForensicsCard({ datasetResult }) {
  const [selectedPoint, setSelectedPoint] = useState(null);

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

        <div className="flex items-center space-x-3">
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
          </div>
        </div>

        <div className="p-3 bg-slate-950/60 border border-slate-800 rounded-xl">
          <div className="text-[10px] text-slate-500 font-sans">Anomalies</div>
          <div className={`text-base font-bold mt-1 ${anomalous_samples_count > 0 ? "text-rose-400" : "text-emerald-400"}`}>
            {anomalous_samples_count}
          </div>
        </div>
      </div>

      {/* 2D Embedding Anomaly Chart */}
      {chart_coordinates.length > 0 && (
        <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between border-b border-slate-800/60 pb-2">
            <div className="flex items-center space-x-2">
              <BarChart2 className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider font-mono">
                2D Feature Space Embedding & Anomaly Scatter Plot
              </span>
            </div>
            <div className="flex items-center space-x-4 text-[11px] font-mono">
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block"></span>
                <span className="text-slate-400">Inlier Sample</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block animate-pulse"></span>
                <span className="text-rose-400 font-semibold">Anomalous Outlier</span>
              </div>
            </div>
          </div>

          <div className="relative w-full h-64 bg-slate-900/40 rounded-lg overflow-hidden border border-slate-800/40 flex items-center justify-center p-4">
            <svg viewBox="-1.2 -1.2 2.4 2.4" className="w-full h-full">
              {/* Grid Lines */}
              <line x1="-1.2" y1="0" x2="1.2" y2="0" stroke="#334155" strokeWidth="0.005" strokeDasharray="0.03" />
              <line x1="0" y1="-1.2" x2="0" y2="1.2" stroke="#334155" strokeWidth="0.005" strokeDasharray="0.03" />
              
              {/* Normal Points */}
              {chart_coordinates.filter(p => !p.is_anomaly).map((pt, idx) => (
                <circle
                  key={`norm-${idx}`}
                  cx={pt.x}
                  cy={-pt.y}
                  r="0.025"
                  className="fill-cyan-400 opacity-70 hover:opacity-100 hover:r-0.04 transition-all cursor-pointer"
                  onMouseEnter={() => setSelectedPoint(pt)}
                />
              ))}

              {/* Anomalous Points */}
              {chart_coordinates.filter(p => p.is_anomaly).map((pt, idx) => (
                <g key={`anom-${idx}`} onMouseEnter={() => setSelectedPoint(pt)}>
                  <circle cx={pt.x} cy={-pt.y} r="0.06" className="fill-rose-500/30 animate-ping" />
                  <circle cx={pt.x} cy={-pt.y} r="0.04" className="fill-rose-500 stroke-white stroke-1 cursor-pointer" />
                  <text x={pt.x + 0.05} y={-pt.y + 0.03} fontSize="0.06" fill="#f43f5e" fontFamily="monospace" fontWeight="bold">
                    #{pt.sample_idx}
                  </text>
                </g>
              ))}
            </svg>

            {/* Hover Tooltip */}
            {selectedPoint && (
              <div className="absolute top-3 right-3 bg-slate-950/90 border border-slate-700 rounded-lg p-2.5 text-[10px] font-mono text-slate-200 shadow-xl backdrop-blur-md">
                <div><span className="text-slate-400">Sample Index:</span> #{selectedPoint.sample_idx}</div>
                <div><span className="text-slate-400">Decision Score:</span> {selectedPoint.score}</div>
                <div><span className="text-slate-400">Classification:</span> <span className={selectedPoint.is_anomaly ? "text-rose-400 font-bold" : "text-emerald-400"}>{selectedPoint.is_anomaly ? "ANOMALOUS OUTLIER" : "NORMAL INLIER"}</span></div>
              </div>
            )}
          </div>
        </div>
      )}

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

      {/* Duplicate Pairs Table */}
      {duplicate_pairs.length > 0 && (
        <div className="space-y-3 border-t border-slate-800 pt-4 font-mono text-xs">
          <div className="flex items-center space-x-2 font-bold text-amber-400">
            <Tag className="w-4 h-4" />
            <span>Identified Duplicate & Near-Duplicate Pairs ({duplicate_pairs.length})</span>
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
                {duplicate_pairs.slice(0, 10).map((pair, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40 transition">
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
        </div>
      )}

      {/* Distribution Summary */}
      {class_channel_distribution.mean_pixel !== undefined && (
        <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-2 font-mono text-xs">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">
            Dataset Feature Pixel Distribution Statistics
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-slate-300 pt-1">
            <div><span className="text-slate-500">Pixel Mean:</span> {class_channel_distribution.mean_pixel}</div>
            <div><span className="text-slate-500">Pixel Std:</span> {class_channel_distribution.std_pixel}</div>
            <div><span className="text-slate-500">Pixel Min:</span> {class_channel_distribution.min_pixel}</div>
            <div><span className="text-slate-500">Pixel Max:</span> {class_channel_distribution.max_pixel}</div>
          </div>
        </div>
      )}

      {/* Honest Limitations Disclaimer Box */}
      {limitations_disclaimer && (
        <div className="p-4 bg-slate-950 border border-slate-800/80 rounded-xl text-xs text-slate-400 space-y-1">
          <div className="flex items-center space-x-1.5 font-semibold text-slate-300">
            <ShieldAlert className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>Honest Assurance & Limitations Disclaimer</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-400">{limitations_disclaimer}</p>
        </div>
      )}
    </div>
  );
}
