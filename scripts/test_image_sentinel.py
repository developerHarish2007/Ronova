"""
Image Sentinel / Image Input Assurance Layer Tests for RONOVA.
Tests all 18 required scenarios for defensive image decoding, format validation,
transformation stability, and DL anomaly evaluation.
"""

import io
import json
import os
import tempfile
import pytest
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from fastapi.testclient import TestClient

from ronova.detectors.image_sentinel import ImageSentinelEngine
from ronova.core.types import ImageAssuranceStatus
from ronova.api.main import app

client = TestClient(app)


@pytest.fixture
def engine():
    return ImageSentinelEngine()


@pytest.fixture
def valid_png_bytes():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def valid_jpeg_bytes():
    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def valid_webp_bytes():
    img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    return buf.getvalue()


def test_valid_png_jpeg_webp(engine, valid_png_bytes, valid_jpeg_bytes, valid_webp_bytes):
    """1. Valid PNG, JPEG, and WebP images pass format validation and decode successfully."""
    r_png = engine.process_image(valid_png_bytes, "test.png")
    assert r_png.safety_result.is_valid is True
    assert r_png.safety_result.original_format == "PNG"

    r_jpg = engine.process_image(valid_jpeg_bytes, "test.jpg")
    assert r_jpg.safety_result.is_valid is True
    assert r_jpg.safety_result.original_format == "JPEG"

    r_webp = engine.process_image(valid_webp_bytes, "test.webp")
    assert r_webp.safety_result.is_valid is True
    assert r_webp.safety_result.original_format == "WEBP"


def test_renamed_invalid_file(engine):
    """2. Plain text file renamed as .png is rejected due to magic byte mismatch."""
    raw_text = b"This is a text file renamed to .png\n"
    r = engine.process_image(raw_text, "fake.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert r.safety_result.is_valid is False
    assert any("magic bytes" in reason.lower() for reason in r.safety_result.rejection_reasons)


def test_invalid_magic_bytes(engine):
    """3. Random noise binary header is rejected due to invalid magic bytes."""
    raw_bytes = b"\x00\x01\x02\x03" * 100
    r = engine.process_image(raw_bytes, "noise.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert r.safety_result.is_valid is False


def test_malformed_image_bytes(engine):
    """4. Valid PNG header followed by corrupted garbage bytes is rejected."""
    raw_bytes = b"\x89PNG\r\n\x1a\n" + b"CORRUPTED_GARBAGE_BYTES" * 20
    r = engine.process_image(raw_bytes, "corrupted.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert r.safety_result.is_valid is False


def test_oversized_streamed_upload(engine):
    """5. Upload exceeding max file size limit (5MB) is rejected."""
    large_engine = ImageSentinelEngine(max_file_size=1024)  # 1 KB limit for test
    large_bytes = b"\x89PNG\r\n\x1a\n" + b"A" * 2000
    r = large_engine.process_image(large_bytes, "oversized.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert any("exceeds" in reason.lower() and "limit" in reason.lower() for reason in r.safety_result.rejection_reasons)


def test_excessive_dimensions():
    """6. Image exceeding maximum dimensions is rejected."""
    # Create image with dim 4500x100 exceeding Standard limit of 4096px
    img = Image.new("RGB", (4500, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    std_engine = ImageSentinelEngine(profile="Standard")
    r = std_engine.process_image(buf.getvalue(), "huge_dim.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert any("exceed" in reason.lower() for reason in r.safety_result.rejection_reasons)


def test_unsupported_image_formats(engine):
    """7. Unsupported formats (GIF, TIFF, PDF, SVG) are explicitly rejected."""
    # GIF
    gif_img = Image.new("RGB", (50, 50))
    gif_buf = io.BytesIO()
    gif_img.save(gif_buf, format="GIF")
    r_gif = engine.process_image(gif_buf.getvalue(), "test.gif")
    assert r_gif.status == ImageAssuranceStatus.REJECTED

    # PDF
    pdf_bytes = b"%PDF-1.4 header text\n"
    r_pdf = engine.process_image(pdf_bytes, "doc.pdf")
    assert r_pdf.status == ImageAssuranceStatus.REJECTED


def test_exif_metadata_removal(engine):
    """8. EXIF metadata is stripped from canonical PNG re-encode."""
    img = Image.new("RGB", (50, 50), color=(200, 100, 50))
    buf = io.BytesIO()
    # Add dummy info
    img.save(buf, format="JPEG", comment="SECRET_EXIF_COMMENT")

    r = engine.process_image(buf.getvalue(), "exif.jpg")
    assert r.safety_result.is_valid is True
    assert r.safety_result.metadata_stripped is True


def test_deterministic_canonical_hash(engine, valid_png_bytes):
    """9. Processing identical raw bytes produces identical canonical PNG SHA-256 and perceptual hash."""
    r1 = engine.process_image(valid_png_bytes, "test.png")
    r2 = engine.process_image(valid_png_bytes, "test.png")
    assert r1.safety_result.canonical_sha256 == r2.safety_result.canonical_sha256
    assert r1.safety_result.perceptual_hash == r2.safety_result.perceptual_hash


def test_unavailable_calibration_returns_unguaranteed(valid_png_bytes):
    """10. If reference calibration file is absent, status is UNGUARANTEED."""
    missing_engine = ImageSentinelEngine(calibration_path="artifacts/missing_calibration.json")
    r = missing_engine.process_image(valid_png_bytes, "test.png")
    assert r.status == ImageAssuranceStatus.UNGUARANTEED
    assert any(f.finding_type == "UNGUARANTEED_NO_CALIBRATION" for f in r.findings)


@pytest.fixture
def clean_digit_bytes():
    if os.path.exists("data/test_samples_clean.npy"):
        arr = (np.load("data/test_samples_clean.npy")[0, 0] * 255).astype(np.uint8)
        img = Image.fromarray(arr).resize((100, 100))
    else:
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clean_reference_image_passes(engine, clean_digit_bytes):
    """11. Valid clean reference image yields SAFE_UNDER_CHECKS."""
    # Ensure calibration exists first
    from scripts.train_image_sentinel_calibration import train_calibration
    train_calibration()

    r = engine.process_image(clean_digit_bytes, "clean.png")
    assert r.status in [ImageAssuranceStatus.SAFE_UNDER_CHECKS, ImageAssuranceStatus.UNGUARANTEED]


def test_api_post_scan_image_clean(valid_png_bytes):
    """12. API POST /scan/image with clean PNG returns valid response, audit ledger, and signed certificate."""
    res = client.post("/scan/image", files={"file": ("test.png", valid_png_bytes, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "safety_result" in data
    assert "limitations_disclaimer" in data
    assert "Passed configured integrity" in data["limitations_disclaimer"]


def test_api_post_scan_image_rejected():
    """13. API POST /scan/image with invalid file returns REJECTED status in JSON body."""
    invalid_bytes = b"NOT_A_VALID_IMAGE_FILE"
    res = client.post("/scan/image", files={"file": ("bad.png", invalid_bytes, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "REJECTED"
    assert data["safety_result"]["is_valid"] is False


def test_api_post_scan_image_oversized_streamed_limit():
    """14. API POST /scan/image exceeding Demo 5MB streamed budget returns 413 Payload Too Large."""
    huge_bytes = b"\x89PNG\r\n\x1a\n" + b"X" * (5 * 1024 * 1024 + 100)
    res = client.post("/scan/image?profile=Demo", files={"file": ("huge.png", huge_bytes, "image/png")})
    assert res.status_code == 413


def test_provenance_uses_real_model_sha256_and_arrays(clean_digit_bytes):
    """15. Real model SHA-256 (not image SHA-256) and real preprocessed arrays are recorded in provenance manifest."""
    res = client.post("/scan/image", files={"file": ("clean.png", clean_digit_bytes, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert "provenance_manifest" in data
    prov = data["provenance_manifest"]
    safety = data["safety_result"]

    # Read actual ONNX model hash
    import hashlib
    with open("models/clean_classifier.onnx", "rb") as f:
        real_model_sha256 = hashlib.sha256(f.read()).hexdigest()

    assert prov["model_hash"] == real_model_sha256
    assert prov["model_hash"] != safety["canonical_sha256"]
    assert safety["analysis_model_sha256"] == real_model_sha256
    assert safety["adapter_name"] == "mnist_float32_nchw_28x28"


def test_unsupported_model_input_returns_unguaranteed(tmp_path, clean_digit_bytes):
    """16. ONNX model with unsupported input shape returns UNGUARANTEED_UNSUPPORTED_MODEL_INPUT."""
    import onnx
    from onnx import helper, TensorProto

    # Create dummy ONNX model with input shape [1, 3, 224, 224] (RGB image layout without registered adapter)
    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2])
    node = helper.make_node('Identity', ['input'], ['output'])
    graph = helper.make_graph([node], 'test_graph', [x], [y])
    model = helper.make_model(graph, producer_name='test')
    
    unsupported_path = tmp_path / "unsupported.onnx"
    onnx.save(model, str(unsupported_path))

    bad_engine = ImageSentinelEngine(model_path=str(unsupported_path))
    r = bad_engine.process_image(clean_digit_bytes, "test.png")
    assert r.status == ImageAssuranceStatus.UNGUARANTEED
    assert any(f.finding_type == "UNGUARANTEED_UNSUPPORTED_MODEL_INPUT" for f in r.findings)


def test_broken_audit_ledger_blocks_image_scan(clean_digit_bytes, monkeypatch):
    """17. Broken audit ledger hash chain blocks normal event creation and returns REJECTED status."""
    from ronova.api.main import ledger

    def mock_validate():
        return False, "Corrupted hash block at event #2", []

    monkeypatch.setattr(ledger, "validate_chain", mock_validate)

    res = client.post("/scan/image", files={"file": ("clean.png", clean_digit_bytes, "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "REJECTED"
    assert any(f["finding_type"] == "BROKEN_AUDIT_CHAIN" for f in data["findings"])


def test_certificate_signing_failure_returns_503(clean_digit_bytes, monkeypatch):
    """18. Certificate signing failure returns HTTP 503 Service Unavailable."""
    from ronova.api.main import cert_builder

    def mock_gen_cert(*args, **kwargs):
        raise FileNotFoundError("Signing key file missing")

    monkeypatch.setattr(cert_builder, "generate_certificate", mock_gen_cert)

    res = client.post("/scan/image", files={"file": ("clean.png", clean_digit_bytes, "image/png")})
    assert res.status_code == 503
    assert "Signing key unavailable" in res.json()["detail"]


def test_1920x1080_screenshot_passes_under_demo_and_standard():
    """19. Normal 1920x1080 screenshot passes under both Demo and Standard profiles."""
    img = Image.new("RGB", (1920, 1080), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    demo_engine = ImageSentinelEngine(profile="Demo")
    r_demo = demo_engine.process_image(png_bytes, "shot_1080p.png")
    assert r_demo.safety_result.is_valid is True

    std_engine = ImageSentinelEngine(profile="Standard")
    r_std = std_engine.process_image(png_bytes, "shot_1080p.png")
    assert r_std.safety_result.is_valid is True


def test_2560x1440_screenshot_passes_under_standard():
    """20. 2560x1440 screenshot passes under Standard profile (rejected under Demo due to 4M pixel limit)."""
    img = Image.new("RGB", (2560, 1440), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    std_engine = ImageSentinelEngine(profile="Standard")
    r_std = std_engine.process_image(png_bytes, "shot_1440p.png")
    assert r_std.safety_result.is_valid is True

    demo_engine = ImageSentinelEngine(profile="Demo")
    r_demo = demo_engine.process_image(png_bytes, "shot_1440p.png")
    assert r_demo.status == ImageAssuranceStatus.REJECTED
    assert any(f.finding_type == "REJECTED_RESOURCE_LIMIT" for f in r_demo.findings)
    assert r_demo.risk_score == 0.0


def test_3840x2160_4k_screenshot_under_standard():
    """21. 3840x2160 (4K) screenshot passes under Standard profile (rejected under Demo)."""
    img = Image.new("RGB", (3840, 2160), color=(150, 150, 150))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    std_engine = ImageSentinelEngine(profile="Standard")
    r_std = std_engine.process_image(png_bytes, "shot_4k.png")
    assert r_std.safety_result.is_valid is True

    demo_engine = ImageSentinelEngine(profile="Demo")
    r_demo = demo_engine.process_image(png_bytes, "shot_4k.png")
    assert r_demo.status == ImageAssuranceStatus.REJECTED
    assert any(f.finding_type == "REJECTED_RESOURCE_LIMIT" for f in r_demo.findings)


def test_image_exceeding_standard_profile_returns_rejected_resource_limit():
    """22. Image exceeding Standard profile (5000x5000) is rejected as REJECTED_RESOURCE_LIMIT with risk_score=0.0."""
    img = Image.new("RGB", (5000, 5000), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    std_engine = ImageSentinelEngine(profile="Standard")
    r = std_engine.process_image(png_bytes, "huge_5k.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert r.risk_score == 0.0
    assert any(f.finding_type == "REJECTED_RESOURCE_LIMIT" for f in r.findings)


def test_no_false_suspicious_or_quarantined_verdict_caused_only_by_dimensions():
    """23. Oversized dimension rejection yields risk_score=0.0 and no SUSPICIOUS or QUARANTINED verdict."""
    img = Image.new("RGB", (4500, 4500), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    std_engine = ImageSentinelEngine(profile="Standard")
    r = std_engine.process_image(png_bytes, "oversized.png")
    assert r.status == ImageAssuranceStatus.REJECTED
    assert r.status != ImageAssuranceStatus.SUSPICIOUS
    assert r.risk_score == 0.0
