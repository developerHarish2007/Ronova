import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Gauge, Info, AlertTriangle, CheckCircle } from "lucide-react";

export default function DriftCard({ inputDriftData }) {
  if (!inputDriftData) {
    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
              <Gauge className="w-6 h-6" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-100">Input Distribution Drift</h3>
              <p className="text-xs text-slate-400 font-mono">
                Operations Assurance & Feature-Space Kolmogorov-Smirnov / PSI Analysis
              </p>
            </div>
          </div>

          <div className="px-3 py-1.5 bg-slate-950 border border-slate-800 rounded-lg text-xs font-mono text-slate-400">
            Awaiting Batch Input Scan
          </div>
        </div>

        <p className="text-xs text-slate-400 leading-relaxed font-sans">
          Upload an input batch via <code className="text-cyan-400 font-mono">POST /check/input</code> or run a model scan to evaluate live Input Distribution Drift.
        </p>
      </div>
    );
  }

  const { input_distribution_drift_score, method, status, ref_means, live_means } = inputDriftData;

  const isNormal = status === "NORMAL";
  const isElevated = status === "ELEVATED";
  const isHigh = status === "HIGH";

  // Build chart data comparing ref vs live feature embedding means
  const chartData = (ref_means || []).map((refVal, idx) => ({
    dimension: `Dim ${idx + 1}`,
    reference: Number(refVal.toFixed(3)),
    live: Number((live_means?.[idx] || 0).toFixed(3)),
  }));

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <Gauge className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">Input Distribution Drift</h3>
            <p className="text-xs text-slate-400 font-mono">
              Operations Assurance & Feature-Space Kolmogorov-Smirnov / PSI Analysis
            </p>
          </div>
        </div>

        <div className={`px-3.5 py-1.5 rounded-xl border font-mono text-xs font-bold flex items-center space-x-2 ${
          isNormal ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" :
          isElevated ? "bg-amber-500/10 border-amber-500/30 text-amber-400" :
          "bg-rose-500/10 border-rose-500/30 text-rose-400"
        }`}>
          {isNormal && <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />}
          {(isElevated || isHigh) && <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />}
          <span>STATUS: {status}</span>
        </div>
      </div>

      {/* Metrics Banner */}
      <div className="grid grid-cols-3 gap-3 font-mono text-xs">
        <div className="bg-slate-950/70 border border-slate-800 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">Input Distribution Drift Score</div>
          <div className={`text-xl font-bold mt-0.5 ${isNormal ? "text-emerald-400" : "text-amber-400"}`}>
            {input_distribution_drift_score?.toFixed(4)}
          </div>
        </div>
        <div className="bg-slate-950/70 border border-slate-800 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">Statistical Test Method</div>
          <div className="text-xl font-bold text-cyan-400 mt-0.5">
            2-Sample {method || "KS"} Test
          </div>
        </div>
        <div className="bg-slate-950/70 border border-slate-800 p-3 rounded-xl">
          <div className="text-[10px] text-slate-500 uppercase">Drift Threshold Boundary</div>
          <div className="text-sm font-semibold text-slate-300 mt-1">
            Normal &lt; 0.25 | High &ge; 0.50
          </div>
        </div>
      </div>

      {/* Recharts Comparison Bar Chart */}
      {chartData.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-slate-400 font-sans">
            Embedding Dimension Means: Reference Training Set vs Live Batch
          </div>

          <div className="h-56 w-full bg-slate-950/80 border border-slate-800 rounded-xl p-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <XAxis dataKey="dimension" tick={{ fill: "#64748b", fontSize: 10 }} />
                <YAxis tick={{ fill: "#64748b", fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "0.5rem", fontSize: "12px" }}
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "5px" }} />
                <Bar dataKey="reference" fill="#06b6d4" name="Reference Baseline" radius={[4, 4, 0, 0]} />
                <Bar dataKey="live" fill={isNormal ? "#10b981" : "#f43f5e"} name="Live Input Batch" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Methodology Limitation Notice */}
      <div className="bg-slate-950/60 border border-slate-800/80 rounded-xl p-3.5 flex items-start space-x-3 text-xs text-slate-400 font-sans">
        <Info className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="font-semibold text-slate-300">Methodology & Limitations</div>
          <p className="text-[11px] text-slate-400 leading-relaxed">
            Input Distribution Drift measures statistical covariate shift between reference baseline training embeddings and live operational batches. 
            <span className="text-slate-300 font-medium font-mono"> It does not claim complete adversarial example detection or universal out-of-distribution (OOD) guarantees.</span>
          </p>
        </div>
      </div>
    </div>
  );
}
