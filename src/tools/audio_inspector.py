#!/usr/bin/env python3
"""
AUDIO BINARY FORMAT ANALYZER & DIAGNOSTIC TOOL
============================================
Diagnoses periodic clicks, pop noises, and file structure anomalies.

Features:
- Detects massive signal jumps (artifacts).
- "Debounces" detection to avoid counting the same click multiple times.
- Analyzes periodicity to identify block-write issues (e.g., 64KB vs 128KB).
- Hex Inspector to visualize the exact bytes causing the glitch.

Usage:
    python tools/audio_inspector.py ./data/raw/file.BIN --show 6
"""

import argparse
import numpy as np
import os
import sys

def analyze_audio(filepath, header_size=144, threshold=15000, context_bytes=24, show_count=6):
    """
    Analyzes a raw binary audio file for discontinuities and periodic artifacts.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found: {filepath}")
        return

    filename = os.path.basename(filepath)
    print(f"\n{'='*20} DIAGNOSTIC REPORT: {filename} {'='*20}")
    print(f"File Size:   {os.path.getsize(filepath):,} bytes")
    print(f"Header Skip: {header_size} bytes")
    print(f"Threshold:   {threshold} (amplitude jump)")
    print("-" * 60)

    try:
        # 1. Load Data
        with open(filepath, 'rb') as f:
            # Read full raw bytes for Hex Context later
            full_raw_bytes = f.read()

        # Isolate Payload (Skip Header)
        payload_bytes = full_raw_bytes[header_size:]

        # Interpret as Signed 16-bit PCM (Little Endian)
        # Using frombuffer is much faster than fromfile for memory objects
        audio_samples = np.frombuffer(payload_bytes, dtype='<i2')

        print(f"Total Samples: {len(audio_samples):,}")
        print(f"Duration:      {len(audio_samples) / 48000:.2f} seconds (assuming 48kHz)")

        # 2. Detect Discontinuities (Derivative)
        # Calculate the difference between adjacent samples
        diffs = np.diff(audio_samples.astype(np.int32))

        # Find raw indices where the jump is massive (larger than threshold)
        raw_indices = np.where(np.abs(diffs) > threshold)[0]

        # --- DEBOUNCE LOGIC ---
        # The detector often triggers twice on one artifact (Entry & Exit).
        # We group clicks that happen within 100 samples of each other.
        click_indices = []
        if len(raw_indices) > 0:
            click_indices.append(raw_indices[0])
            for idx in raw_indices[1:]:
                # If this click is more than 100 samples away from the last one, it's a new event
                if idx - click_indices[-1] > 100:
                    click_indices.append(idx)

        print(f"\n[1] ARTIFACT DETECTION")
        print(f"    Found {len(click_indices)} distinct events (Filtered from {len(raw_indices)} raw jumps).")

        if len(click_indices) == 0:
            print("   [SUCCESS] No major artifacts found. Signal seems continuous.")
            return

        # 3. Analyze Periodicity
        print(f"\n[2] PERIODICITY ANALYSIS (First {show_count} Intervals)")

        distances = []
        # Show intervals between the artifacts we are about to display
        for i in range(min(show_count, len(click_indices) - 1)):
            idx_current = click_indices[i]
            idx_next = click_indices[i+1]

            dist_samples = idx_next - idx_current
            dist_bytes = dist_samples * 2  # 16-bit = 2 bytes
            distances.append(dist_bytes)

            time_diff = dist_samples / 48000.0

            print(f"    Event #{i+1} -> #{i+2}:  {dist_bytes} bytes  ({dist_samples} samples, {time_diff:.3f}s)")

        # Pattern Recognition (Block Logic)
        if len(distances) > 0:
            avg_dist = np.mean(distances)
            print(f"\n    -> Average Distance: {avg_dist:.1f} bytes")

            # Check for common "SD card" page sizes
            if 65500 < avg_dist < 65600:
                 print("   [WARNING] DIAGNOSIS: Confirmed 64KB (65536 byte) Page Artifacts.")
            elif 131000 < avg_dist < 132000:
                print("   [WARNING] DIAGNOSIS: Strong indicator of 128KB Block Artifacts.")

        # 4. Hex Inspection (The "Smoking Gun")
        print(f"\n[3] HEX INSPECTOR (Showing first {show_count} distinct artifacts)")

        for i in range(min(show_count, len(click_indices))):
            idx = click_indices[i]

            # Calculate absolute byte offset in file (including header)
            # Sample Index * 2 bytes/sample + Header Size
            abs_byte_offset = (idx * 2) + header_size

            # Grab context window
            start_b = max(0, abs_byte_offset - context_bytes)
            end_b = min(len(full_raw_bytes), abs_byte_offset + context_bytes)

            snippet = full_raw_bytes[start_b:end_b]

            print(f"\n    Event #{i+1} at Absolute Byte {abs_byte_offset}:")

            # Format Hex nicely
            hex_str = snippet.hex(' ').upper()

            # Try to align the pointer roughly to the center
            # Each byte is 3 chars ("XX ")
            # We calculate how far into the snippet our target byte is
            offset_in_snippet = abs_byte_offset - start_b
            pointer_pos = offset_in_snippet * 3

            print(f"    HEX: {hex_str}")
            print(f"         {' ' * pointer_pos}^^ CLICK HERE")

            # ASCII Decode (to see if 'IMU' or 'M' or timestamps appear)
            try:
                ascii_repr = "".join([chr(b) if 32 <= b <= 126 else '.' for b in snippet])
                print(f"    TXT: {ascii_repr}")
            except:
                pass

        print(f"\n{'='*60}\n")

    except Exception as e:
        print(f"[ERROR] Critical Failure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze binary audio files for glitches.")
    parser.add_argument("path", help="Path to the .BIN file")
    parser.add_argument("--header", type=int, default=144, help="Header size to skip (default: 144)")
    parser.add_argument("--threshold", type=int, default=15000, help="Signal jump threshold (default: 15000)")
    parser.add_argument("--show", type=int, default=6, help="Number of artifacts to display (default: 6)")

    args = parser.parse_args()

    # We define the context window size here (fixed to 24 bytes for good visibility)
    CONTEXT_BYTES = 24

    analyze_audio(
        args.path,
        header_size=args.header,
        threshold=args.threshold,
        context_bytes=CONTEXT_BYTES,
        show_count=args.show
    )
