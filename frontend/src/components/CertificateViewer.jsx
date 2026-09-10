import React, { useState } from "react";
import { QRCodeSVG } from "qrcode.react";
import { FileJson, Download, ShieldCheck, QrCode, RefreshCw } from "lucide-react";

export default function CertificateViewer({ result }) {
  if (!result) return null;

  const [verifying, setVerifying] = useState(false);
  const [verifyStatus, setVerifyStatus] = useState(
    result.signed_checkpoint?.verification_status || "VALID"
  );

  const certData = result.canonical_certificate || result;
  const qrPayload = certData.qr_payload || {
    spec_version: "RONOVA-QR-v1",
    signer_identity: "RONOVA-ROOT-ED25519",
    cert_hash: certData.certificate_header?.cert_hash || "",
    model_sha256: result.sha256 || "",
    verdict: result.policy?.verdict || "UNKNOWN",
    risk_score: result.policy?.risk_score || 0,
    timestamp: Date.now() / 1000,
    canonical_payload: certData.canonical_payload || {},
    signature: certData.signature || result.signed_checkpoint?.signature || "",
  };
  const qrDataStr = JSON.stringify(qrPayload);

  const handleVerify = async () => {
    setVerifying(true);
    try {
      const checkpoint = result.signed_checkpoint?.checkpoint_payload || {};
      const sig = result.signed_checkpoint?.signature || "";

      const params = new URLSearchParams({
        signature: sig,
        event_hash: checkpoint.event_hash || "",
        model_sha256: checkpoint.model_sha256 || result.sha256 || "",
        timestamp: String(checkpoint.timestamp || ""),
        previous_event_hash: checkpoint.previous_event_hash || ("0" * 64),
      });

      const res = await fetch(`http://localhost:8000/verify/certificate?${params.toString()}`);
      if (res.ok) {
        const data = await res.json();
        setVerifyStatus(data.verification_status);
      }
    } catch (err) {
      setVerifyStatus("INVALID");
    } finally {
      setVerifying(false);
    }
  };

  const handleDownloadJSON = () => {
    const jsonStr = JSON.stringify(certData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ronova_assurance_cert_${(result.sha256 || "model").slice(0, 8)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 shadow-2xl space-y-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-cyan-500/10 border border-cyan-500/30 rounded-xl text-cyan-400">
            <FileJson className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-100">Signed Assurance Certificate</h3>
            <p className="text-xs text-slate-400 font-mono">
              Offline Verifiable Cryptographic Assurance Record & Standalone Verifier
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-3 font-mono text-xs">
          <button
            onClick={handleVerify}
            disabled={verifying}
            className="px-3 py-2 bg-slate-950 hover:bg-slate-800 border border-slate-700 text-cyan-400 font-semibold rounded-xl transition flex items-center space-x-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${verifying ? "animate-spin" : ""}`} />
            <span>Verify Signature</span>
          </button>

          <button
            onClick={handleDownloadJSON}
            className="px-3.5 py-2 bg-cyan-600 hover:bg-cyan-500 text-white font-semibold rounded-xl shadow transition flex items-center space-x-1.5"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* Verification Status & Live QR Code */}
      <div className="grid grid-cols-3 gap-4 font-sans text-xs">
        <div className="col-span-2 bg-slate-950/80 border border-slate-800 p-5 rounded-xl flex items-center space-x-4">
          <ShieldCheck className="w-8 h-8 text-emerald-400 shrink-0" />
          <div className="space-y-1">
            <div className="text-[10px] text-slate-400 uppercase font-mono">Ed25519 Trust Root Verification</div>
            <div className="text-base font-bold text-emerald-400 font-mono">
              Signature Status: {verifyStatus}
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed font-sans">
              Signer: <span className="font-mono text-slate-300">RONOVA-ROOT-ED25519</span>. You can verify this certificate independently offline using:
              <code className="block mt-1 p-2 bg-slate-900 border border-slate-800 rounded font-mono text-[10px] text-cyan-400">
                python verify_certificate.py cert.json trust/root_public.pem
              </code>
              <span className="block mt-1 text-[10px] text-slate-500 italic">
                Note: Cryptographic verification proves certificate authenticity & payload integrity, not model safety.
              </span>
            </p>
          </div>
        </div>

        {/* Live QR Code Box */}
        <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl flex flex-col items-center justify-center space-y-2 text-center">
          <div className="p-2 bg-white rounded-xl shadow-lg">
            <QRCodeSVG value={qrDataStr} size={112} level="L" />
          </div>
          <div className="text-[10px] font-mono text-slate-400">
            Independent Offline QR Payload
          </div>
        </div>
      </div>

      {/* JSON Viewer */}
      <div className="space-y-2">
        <div className="text-xs font-semibold text-slate-300">Canonical JSON Certificate Source of Truth</div>
        <pre className="p-4 bg-slate-950 border border-slate-800 rounded-xl text-[11px] font-mono text-emerald-400/90 overflow-x-auto max-h-80">
          {JSON.stringify(certData, null, 2)}
        </pre>
      </div>
    </div>
  );
}
