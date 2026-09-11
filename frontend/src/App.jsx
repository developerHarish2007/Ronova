import React, { useState } from "react";
import { Shield, FileCheck, Cpu, Database, Image as ImageIcon, Lock, RefreshCw, AlertTriangle } from "lucide-react";
import VerdictPanel from "./components/VerdictPanel";
import PerturbationViz from "./components/PerturbationViz";
import ProvenanceGraph from "./components/ProvenanceGraph";
import DriftCard from "./components/DriftCard";
import CertificateViewer from "./components/CertificateViewer";
import ImageSentinelCard from "./components/ImageSentinelCard";
import DatasetForensicsCard from "./components/DatasetForensicsCard";
import "./App.css";

const THEME_OPTIONS = [
  { id: "cyber", label: "Cyber", dot: "theme-dot-cyber" },
  { id: "mono", label: "Mono", dot: "theme-dot-mono" },
  { id: "white", label: "White", dot: "theme-dot-white" },
  { id: "silk", label: "Silk", dot: "theme-dot-silk" },
];

const TAB_OPTIONS = [
  { id: "model", label: "Model Assurance (POST /scan/model)", icon: Cpu },
  { id: "dataset", label: "Dataset Forensics (POST /scan/dataset)", icon: Database },
  { id: "image", label: "Image Sentinel (POST /scan/image)", icon: ImageIcon },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("model");
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [datasetResult, setDatasetResult] = useState(null);
  const [imageResult, setImageResult] = useState(null);
  const [resourceProfile, setResourceProfile] = useState("Standard");
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("ronova-theme") || "cyber");

  React.useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("ronova-theme", theme);
  }, [theme]);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setDatasetResult(null);
      setImageResult(null);
      setError(null);
    }
  };

  const handleScanModel = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/scan/model", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Model scan request failed");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleScanDataset = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/scan/dataset", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Dataset scan request failed");
      }

      const data = await response.json();
      setDatasetResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleScanImage = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`http://localhost:8000/scan/image?profile=${resourceProfile}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Image scan request failed");
      }

      const data = await response.json();
      setImageResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-start p-6 space-y-6">
      {/* Top Console Bar */}
      <header className="top-console max-w-6xl w-full flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-2xl text-cyan-400">
            <Shield className="w-8 h-8" />
          </div>
          <div>
            <h1 className="text-2xl font-black tracking-tight text-white">RONOVA Console</h1>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Robust Offline Network for Observation, Verification & Assurance
            </p>
          </div>
        </div>

        <div className="top-console-actions flex items-center space-x-3">
          <div className="theme-control" title="Change console appearance">
            <div className="theme-switcher" role="group" aria-label="Console theme selector">
              <span
                className="theme-active-indicator"
                aria-hidden="true"
                style={{ "--theme-index": Math.max(0, THEME_OPTIONS.findIndex((item) => item.id === theme)) }}
              />
              {THEME_OPTIONS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`theme-button ${theme === item.id ? "active" : ""}`}
                  onClick={() => setTheme(item.id)}
                  aria-pressed={theme === item.id}
                  title={`Use ${item.label} theme`}
                >
                  <span className={`theme-dot ${item.dot}`} />
                  <span>{item.label}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center space-x-2 text-xs font-mono bg-slate-900 px-3.5 py-2 rounded-xl border border-slate-800 text-slate-300">
            <Lock className="w-4 h-4 text-emerald-400" />
            <span>Trust Root: Ed25519 Active</span>
          </div>
        </div>
      </header>

      {/* Royal Minimalist Tab Switcher */}
      <div className="mode-switcher-shell max-w-6xl w-full relative flex p-1.5 rounded-2xl border font-sans text-xs">
        <span
          className="tab-active-indicator"
          aria-hidden="true"
          style={{ "--tab-index": Math.max(0, TAB_OPTIONS.findIndex((t) => t.id === activeTab)) }}
        />
        {TAB_OPTIONS.map((tab) => {
          const IconComp = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => { setActiveTab(tab.id); setSelectedFile(null); setError(null); }}
              className={`mode-tab relative z-10 flex-1 py-3 px-4 rounded-xl font-medium transition-all flex items-center justify-center space-x-2.5 ${
                isActive ? "active" : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <IconComp className={`w-4 h-4 transition-transform duration-300 ${isActive ? "scale-110 text-inherit" : "opacity-75"}`} />
              <span className="tracking-wide font-semibold">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Main Content Area */}
      <main key={activeTab} className="dashboard-main max-w-6xl w-full space-y-6">
        {activeTab === "model" && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-md flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1 mr-4">
                <label className="flex-1 flex items-center space-x-3 border border-slate-800 hover:border-cyan-500/50 bg-slate-950/80 rounded-xl p-3.5 cursor-pointer transition">
                  <input type="file" accept=".onnx" onChange={handleFileChange} className="hidden" />
                  <FileCheck className="w-6 h-6 text-cyan-400 shrink-0" />
                  <div className="truncate">
                    <div className="text-xs font-semibold text-slate-200 truncate">
                      {selectedFile ? selectedFile.name : "Select ONNX Classifier Model"}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : "Choose clean_classifier.onnx or backdoored_classifier.onnx"}
                    </div>
                  </div>
                </label>
              </div>

              <button
                onClick={handleScanModel}
                disabled={!selectedFile || loading}
                className="px-6 py-3.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 text-white font-semibold text-xs rounded-xl shadow-lg transition flex items-center space-x-2"
              >
                {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Cpu className="w-4 h-4" />}
                <span>{loading ? "Running Pipeline..." : "Scan Model Artifact"}</span>
              </button>
            </div>

            {error && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-400 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {result && (
              <div className="space-y-6">
                <VerdictPanel
                  policy={result.policy}
                  sha256={result.sha256}
                  filename={result.filename}
                  manifestVerification={result.manifest_verification}
                />
                <PerturbationViz stripAnalysis={result.strip_analysis} />
                <ProvenanceGraph
                  provenanceManifest={result.provenance_manifest}
                  signedCheckpoint={result.signed_checkpoint}
                />
                <DriftCard inputDriftData={result.input_distribution_drift} />
                <CertificateViewer result={result} />
              </div>
            )}
          </div>
        )}

        {activeTab === "dataset" && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-md flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1 mr-4">
                <label className="flex-1 flex items-center space-x-3 border border-slate-800 hover:border-cyan-500/50 bg-slate-950/80 rounded-xl p-3.5 cursor-pointer transition">
                  <input type="file" accept=".npy,.npz" onChange={handleFileChange} className="hidden" />
                  <FileCheck className="w-6 h-6 text-cyan-400 shrink-0" />
                  <div className="truncate">
                    <div className="text-xs font-semibold text-slate-200 truncate">
                      {selectedFile ? selectedFile.name : "Select Dataset (.npy/.npz)"}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : "Choose clean_dataset.npy, poisoned_dataset.npy, or clean_overlays.npy"}
                    </div>
                  </div>
                </label>
              </div>

              <div className="flex items-center space-x-3">

                <button
                  onClick={handleScanDataset}
                  disabled={!selectedFile || loading}
                  className="px-6 py-3.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 text-white font-semibold text-xs rounded-xl shadow-lg transition flex items-center space-x-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                  <span>{loading ? "Scanning..." : "Scan Dataset Artifact"}</span>
                </button>
              </div>
            </div>

            {error && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-400 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {datasetResult && (
              <DatasetForensicsCard datasetResult={datasetResult} />
            )}
          </div>
        )}

        {activeTab === "image" && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-6 shadow-xl backdrop-blur-md flex items-center justify-between">
              <div className="flex items-center space-x-4 flex-1 mr-4">
                <label className="flex-1 flex items-center space-x-3 border border-slate-800 hover:border-cyan-500/50 bg-slate-950/80 rounded-xl p-3.5 cursor-pointer transition">
                  <input type="file" accept=".png,.jpeg,.jpg,.webp" onChange={handleFileChange} className="hidden" />
                  <ImageIcon className="w-6 h-6 text-cyan-400 shrink-0" />
                  <div className="truncate">
                    <div className="text-xs font-semibold text-slate-200 truncate">
                      {selectedFile ? selectedFile.name : "Select Image Input (PNG, JPEG, WebP)"}
                    </div>
                    <div className="text-[10px] text-slate-500 font-mono">
                      {selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : "Supports PNG, JPEG, WebP format validation & isolated decoding"}
                    </div>
                  </div>
                </label>
              </div>

              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-slate-400 font-semibold">Resource Policy:</span>
                  <select
                    value={resourceProfile}
                    onChange={(e) => setResourceProfile(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-cyan-400 font-bold focus:outline-none focus:border-cyan-500 cursor-pointer"
                  >
                    <option value="Standard">Standard (Max 4096px / 10MB)</option>
                    <option value="Demo">Demo (Max 2048px / 5MB)</option>
                  </select>
                </div>

                <button
                  onClick={handleScanImage}
                  disabled={!selectedFile || loading}
                  className="px-6 py-3.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 text-white font-semibold text-xs rounded-xl shadow-lg transition flex items-center space-x-2"
                >
                  {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ImageIcon className="w-4 h-4" />}
                  <span>{loading ? "Analyzing Image..." : "Scan Image Sentinel"}</span>
                </button>
              </div>
            </div>

            {error && (
              <div className="p-4 bg-rose-500/10 border border-rose-500/30 rounded-xl text-xs text-rose-400 flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {imageResult && (
              <ImageSentinelCard imageResult={imageResult} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
