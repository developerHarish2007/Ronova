import React, { useMemo } from "react";
import ReactFlow, { Background, Controls } from "reactflow";
import "reactflow/dist/style.css";
import { GitCommit, ShieldCheck, Lock } from "lucide-react";

export default function ProvenanceGraph({ provenanceManifest, signedCheckpoint }) {
  if (!provenanceManifest) return null;

  const { nodes, edges } = useMemo(() => {
    const fields = [
      { id: "dataset_hash", label: "Dataset Hash", val: provenanceManifest.dataset_hash, x: 50, y: 50 },
      { id: "model_hash", label: "Model Hash", val: provenanceManifest.model_hash, x: 260, y: 50 },
      { id: "config_hash", label: "Config Hash", val: provenanceManifest.config_hash, x: 470, y: 50 },
      { id: "preprocess_hash", label: "Preprocess Hash", val: provenanceManifest.preprocess_hash, x: 680, y: 50 },

      { id: "runtime_hash", label: "Runtime Hash", val: provenanceManifest.runtime_hash, x: 50, y: 160 },
      { id: "input_hash", label: "Input Hash", val: provenanceManifest.input_hash, x: 260, y: 160 },
      { id: "output_hash", label: "Output Hash", val: provenanceManifest.output_hash, x: 470, y: 160 },
      { id: "assurance_engine_hash", label: "Assurance Engine Hash", val: provenanceManifest.assurance_engine_hash, x: 680, y: 160 },
    ];

    const flowNodes = fields.map((f) => ({
      id: f.id,
      position: { x: f.x, y: f.y },
      data: {
        label: (
          <div className="p-2.5 bg-slate-950/90 border border-slate-800 rounded-xl text-left space-y-1 w-44 shadow-lg font-mono text-[10px]">
            <div className="text-slate-400 font-sans text-[10px] font-semibold">{f.label}</div>
            <div className="text-cyan-400 font-mono truncate">{f.val ? `${f.val.slice(0, 14)}...` : "—"}</div>
          </div>
        ),
      },
      style: { background: "transparent", border: "none", padding: 0 },
    }));

    // Add Checkpoint Node
    flowNodes.push({
      id: "signed_checkpoint",
      position: { x: 360, y: 280 },
      data: {
        label: (
          <div className="p-3 bg-emerald-950/80 border border-emerald-500/50 rounded-2xl text-center space-y-1 w-64 shadow-2xl backdrop-blur-md">
            <div className="flex items-center justify-center space-x-1.5 text-xs font-bold text-emerald-400">
              <ShieldCheck className="w-4 h-4" />
              <span>Signed Ed25519 Checkpoint</span>
            </div>
            <div className="text-[10px] font-mono text-emerald-300">
              Status: {signedCheckpoint?.verification_status || "VALID"}
            </div>
            <div className="text-[9px] font-mono text-slate-400 truncate">
              Sig: {signedCheckpoint?.signature ? `${signedCheckpoint.signature.slice(0, 18)}...` : "—"}
            </div>
          </div>
        ),
      },
      style: { background: "transparent", border: "none", padding: 0 },
    });

    const flowEdges = fields.map((f) => ({
      id: `edge-${f.id}`,
      source: f.id,
      target: "signed_checkpoint",
      animated: true,
      style: { stroke: "#06b6d4", strokeWidth: 1.5 },
    }));

    return { nodes: flowNodes, edges: flowEdges };
  }, [provenanceManifest, signedCheckpoint]);

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-4">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <GitCommit className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">8-Field Provenance Hash DAG</h3>
            <p className="text-xs text-slate-400 font-mono">
              Immutable Cryptographic Audit Graph anchored to Ed25519 Trust Root
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2 text-xs font-mono bg-slate-950 px-3 py-1.5 rounded-lg border border-slate-800 text-slate-300">
          <Lock className="w-3.5 h-3.5 text-emerald-400" />
          <span>Checkpoint Signed</span>
        </div>
      </div>

      <div className="h-96 w-full bg-slate-950 border border-slate-800/80 rounded-xl overflow-hidden relative">
        <ReactFlow nodes={nodes} edges={edges} fitView minZoom={0.6} maxZoom={1.2}>
          <Background color="#1e293b" gap={16} />
          <Controls className="bg-slate-900 border-slate-800 text-slate-300 rounded-lg p-1" />
        </ReactFlow>
      </div>
    </div>
  );
}
