#!/usr/bin/env python3
"""
RONOVA Isolated Image Decoding Worker Subprocess:
Performs fail-closed format validation, magic byte verification, defensive resource limit checks,
EXIF metadata stripping, canonical PNG re-encoding, and hash generation.
Executes in a separate worker process isolated from the main FastAPI server.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageFile, ImageOps, ImageStat, PngImagePlugin

# Enable strict Pillow limit checks
Image.MAX_IMAGE_PIXELS = 4194304  # 2048 x 2048 max pixels limit
ImageFile.LOAD_TRUNCATED_IMAGES = False

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP"}
MAGIC_BYTES_MAP = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"RIFF": "WEBP",
}


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img: Image.Image, hash_size: int = 8) -> str:
    """Computes a 64-bit difference hash (dHash) for perceptual image comparison."""
    # Convert to grayscale and resize to (hash_size + 1, hash_size)
    resized = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = list(resized.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)

    # Convert binary list to hex string
    decimal_val = 0
    for bit in difference:
        decimal_val = (decimal_val << 1) | bit
    return f"{decimal_val:016x}"


def detect_magic_format(raw_header: bytes) -> str:
    if raw_header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if raw_header.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if raw_header.startswith(b"RIFF") and len(raw_header) >= 12 and raw_header[8:12] == b"WEBP":
        return "WEBP"

    # Check for known rejected formats
    if raw_header.startswith(b"GIF87a") or raw_header.startswith(b"GIF89a"):
        return "REJECTED_GIF"
    if raw_header.startswith(b"II*\x00") or raw_header.startswith(b"MM\x00*"):
        return "REJECTED_TIFF"
    if raw_header.startswith(b"%PDF"):
        return "REJECTED_PDF"
    if raw_header.startswith(b"PK\x03\x04") or raw_header.startswith(b"7z\xbc\xaf\x27\x1c"):
        return "REJECTED_ARCHIVE"
    if raw_header.startswith(b"MZ"):
        return "REJECTED_EXECUTABLE"
    if b"<svg" in raw_header.lower() or b"<?xml" in raw_header.lower():
        return "REJECTED_SVG"

    return "UNKNOWN"


def decode_and_sanitize(
    input_path: str,
    canonical_output_path: str,
    max_file_size: int = 10485760,  # 10 MB default for Standard
    max_dim: int = 4096,
    max_pixels: int = 16000000,
    max_frames: int = 1,
    profile_name: str = "Standard",
) -> dict:
    # Dynamically adjust strict Pillow limit checks based on active profile
    Image.MAX_IMAGE_PIXELS = max_pixels
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    rejection_reasons = []
    input_file = Path(input_path).resolve()

    if not input_file.exists():
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": [f"File not found: {input_path}"],
        }

    # 1. Source file size limit check
    file_size = input_file.stat().st_size
    if file_size > max_file_size:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_type": "REJECTED_RESOURCE_LIMIT",
            "rejection_reasons": [f"File size ({file_size} bytes) exceeds active processing budget ({max_file_size} bytes, Profile: {profile_name})"],
        }

    if file_size == 0:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": ["Empty file (0 bytes)"],
        }

    # 2. Magic byte verification
    with open(input_file, "rb") as f:
        header = f.read(512)

    magic_fmt = detect_magic_format(header)
    if magic_fmt.startswith("REJECTED_"):
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": [f"Unsupported format rejected by security policy ({magic_fmt})"],
        }

    if magic_fmt not in SUPPORTED_FORMATS:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": [f"Unsupported or unrecognized magic bytes format ({magic_fmt})"],
        }

    # 3. Pillow Image Header Inspection & Decoding inside isolated worker
    original_sha256 = compute_sha256(str(input_file))

    try:
        with Image.open(input_file) as img:
            img_format = (img.format or magic_fmt).upper()

            if img_format not in SUPPORTED_FORMATS:
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "rejection_reasons": [f"Decoded format '{img_format}' is not in supported list {list(SUPPORTED_FORMATS)}"],
                }

            # Check animated / multi-frame
            n_frames = getattr(img, "n_frames", 1)
            is_animated = getattr(img, "is_animated", False) or n_frames > 1
            if is_animated or n_frames > max_frames:
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "rejection_reasons": [f"Multi-frame/animated image rejected (frames: {n_frames}, max allowed: {max_frames})"],
                }

            # Header dimension inspection (size read directly from header metadata before full pixel load)
            width, height = img.size
            total_pixels = width * height

            if width > max_dim or height > max_dim:
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "rejection_type": "REJECTED_RESOURCE_LIMIT",
                    "rejection_reasons": [f"Dimensions ({width}x{height}) exceed maximum allowed dimension ({max_dim}px, Profile: {profile_name})"],
                }

            if total_pixels > max_pixels:
                return {
                    "is_valid": False,
                    "status": "REJECTED",
                    "rejection_type": "REJECTED_RESOURCE_LIMIT",
                    "rejection_reasons": [f"Total pixel count ({total_pixels}) exceeds maximum pixel budget ({max_pixels} pixels, Profile: {profile_name})"],
                }

            # Verify image pixel data can be fully loaded without error
            img.verify()

    except Image.DecompressionBombError as e:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_type": "REJECTED_RESOURCE_LIMIT",
            "rejection_reasons": [f"Decompression bomb / extreme dimension detected ({str(e)})"],
        }
    except Exception as e:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": [f"Image decoding failed: {str(e)}"],
        }

    # Re-open image to process pixels (verify closed the file handle)
    try:
        with Image.open(input_file) as img:
            # Metadata Removal & Color Mode Normalization
            # Convert to RGB (normalizing RGBA / L / CMYK modes)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                # Composite over white background to avoid transparent pixel artifacts
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.convert("RGBA").split()[3])
                rgb_img = bg
            else:
                rgb_img = img.convert("RGB")

            # Calculate perceptual hash before saving
            p_hash = compute_dhash(rgb_img)

            # Re-encode deterministically to canonical PNG without EXIF/metadata
            canonical_file = Path(canonical_output_path).resolve()
            os.makedirs(canonical_file.parent, exist_ok=True)

            # Save clean PNG with no info/exif payload
            clean_png_info = PngImagePlugin.PngInfo()
            rgb_img.save(str(canonical_file), format="PNG", optimize=True, pnginfo=clean_png_info)

            canonical_sha256 = compute_sha256(str(canonical_file))

            return {
                "is_valid": True,
                "status": "SAFE_UNDER_CHECKS",
                "original_sha256": original_sha256,
                "canonical_sha256": canonical_sha256,
                "perceptual_hash": p_hash,
                "original_format": magic_fmt,
                "canonical_format": "PNG",
                "dimensions": [width, height],
                "color_mode": img.mode,
                "metadata_stripped": True,
                "rejection_reasons": [],
                "decode_worker_metadata": {
                    "worker_process_id": os.getpid(),
                    "pixel_count": total_pixels,
                    "n_frames": n_frames,
                    "normalized_color_mode": "RGB",
                },
                "configured_limits": {
                    "active_profile": profile_name,
                    "max_file_size_bytes": max_file_size,
                    "max_dimension_pixels": max_dim,
                    "max_total_pixels": max_pixels,
                    "max_frames": max_frames,
                    "supported_formats": sorted(list(SUPPORTED_FORMATS)),
                },
                "protocol_version": "ronova_image_sentinel_v1",
            }

    except Exception as e:
        return {
            "is_valid": False,
            "status": "REJECTED",
            "rejection_reasons": [f"Canonical re-encoding failed: {str(e)}"],
        }


def main():
    parser = argparse.ArgumentParser(description="RONOVA Subprocess Image Decode Worker")
    parser.add_argument("--input", required=True, help="Input image file path")
    parser.add_argument("--output", required=True, help="Canonical output PNG file path")
    parser.add_argument("--max-size", type=int, default=10485760, help="Max file size in bytes")
    parser.add_argument("--max-dim", type=int, default=4096, help="Max dimension in pixels")
    parser.add_argument("--max-pixels", type=int, default=16000000, help="Max total pixels")
    parser.add_argument("--max-frames", type=int, default=1, help="Max frames allowed")
    parser.add_argument("--profile", default="Standard", help="Active resource policy profile name")
    parser.add_argument("--result-json", required=True, help="Path to write output JSON result")

    args = parser.parse_args()

    result = decode_and_sanitize(
        input_path=args.input,
        canonical_output_path=args.output,
        max_file_size=args.max_size,
        max_dim=args.max_dim,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
        profile_name=args.profile,
    )

    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    sys.exit(0 if result["is_valid"] else 1)


if __name__ == "__main__":
    main()
