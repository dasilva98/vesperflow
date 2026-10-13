🐒 ⚙️ VesperFlow
=====================
![Version](https://img.shields.io/github/v/release/dasilva98/vesperflow)
[![DPZ Lab][dpz-badge]](https://www.dpz.eu/en/about-us)
![License](https://img.shields.io/badge/License-GPLv3-blue)

[Tool Demo Video](https://youtu.be/wNGPoCBlQts)

A headless, scriptable CLI automation pipeline for decoding and processing multi-modal time-series data from bio-logging tags. 

Designed specifically for machine learning and database ingestion workflows, this tool provides a "One-Click" solution to convert raw binary dumps into analysis-ready formats. By bypassing intermediate disk writes in favor of vectorized in-memory array operations, it natively handles proprietary binary decoding, true-time synchronization, and hardware artifact mitigation.

> **DISCLAIMER** This is an unofficial, independent research tool developed by students at the University of Göttingen for the German Primate Center (DPZ). It is **not** affiliated with A.S.D. (Alexander Schwartz Developments). All product names are property of their respective owners and are used here solely for identification and compatibility purposes.

Key Features
-----------------------
The native parsers utilize a standardized **128-byte File Header** and **16-byte Block Header** architecture to unpack payloads, while automatically mitigating known hardware quirks:

### 1. Signal Processing & Artifact Mitigation
### 🏃‍♂️🧭 IMU (Accelerometer & Gyroscope):
- **True-Time Synchronization:** Derives drift-free, per-packet timestamps using the hardware's 1024-tick sub-second counter via vectorized NumPy operations.
- **Glitch Masking:** Automatically hunts and masks reproducible tag file-split artifacts (e.g., the `-8.75` corrupted gyroscope spike) using pandas `NaN` buffers, protecting downstream ML behavioral classifiers.
- **Dynamic Configuration:** Detects the active bitmask and dynamically outputs **Temperature (°C)** and **Barometric Pressure (hPa)** if extended sensors are active.

### 🎙️ **Audio (Microphone):**
- **16-Byte Block Excisions:** Surgically removes the 16-byte sync markers injected at every 64KB page boundary. This mathematically preserves 16-bit sample alignment across the entire stream.
- **Strict RIFF Compliance:** Writes mathematically precise `.wav` (48kHz) headers using native Python libraries, fixing the 16-byte chunk truncation bug found in proprietary extraction tools that causes ML acoustic libraries to crash.
- **DC-Offset Mitigation:** Mutes the ~17ms (`0x8000`) startup pop caused by microphone hardware initialization.

### 🛰️ **GPS (GNSS Snapshots):**
- **Smart Header Detection:** Seamlessly bypasses the 1,008-byte zero-padded buffer injected during the radio's warm-up phase.
- **16-bit I/Q Word Swap:** Applies a vectorized bitwise swap to the raw Intermediate Frequency (IF) radio snapshots, correcting the sample interleaving for software-defined radio (SDR) solvers.

### 2. Scientific Reporting & Analytics
The `RunReporter` module acts as the single source of truth for pipeline execution, producing deterministic database-ready filenames (`YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS_DEVICEID.ext`) and comprehensive `.txt` report cards:
- **Session Inventory:** Automatically flattens the hardware's 256-file chunk folders into unified tag deployments, calculating the absolute calendar window.
- **True Scientific Yield:** Bypasses intermittent sleep-cycle skew by mathematically calculating yield from the extracted arrays ($N_{samples} / f_s$). 
- **Smart Fault Isolation:** Correctly categorizes files with valid headers but zero payload as **WARNING** (Empty) rather than a failure, protecting the extraction success rate from natural tag shutdowns.

### 3. Integrated GeoTag Pipeline (With Open-Source Roadmap)
- **Automated Wrapping:** Automatically launches the manufacturer's `GeoTag.exe` to decode I/Q snapshots into coordinates, while safely parsing the true Start/End timestamps.
- **Graceful Linux/macOS Fallback:** If the Windows-only executable is missing on a headless server or HPC cluster, the orchestrator logs a warning but continues processing all IMU and Audio streams without crashing.
- **Roadmap:** The decoupled GPS architecture is currently being adapted to pipe `.DAT` snapshots into open-source, cross-platform SDR engines (e.g., SnapperGPS), aiming to fully eliminate proprietary Windows dependencies in future releases.


Build & Quick Start
-----------------------

### 1. Prerequisites

- **Python 3.12+**
- **Cross-Platform Support:** IMU and Audio decoding run flawlessly natively on Windows, Linux, and macOS. 
- **Legacy GPS Support:** Currently, GPS coordinate decoding requires **Windows 10/11** and the presence of `GeoTag.exe` and `GeoTagEngine.exe` (placed in `external_tools/CG/GeoTag/` and `external_tools/CG/GeoTagEngine/`).

### 2. Installation

Clone the repository:

    git clone https://github.com/dasilva98/vesperflow
    cd vesperflow

Set up the virtual environment:

    # Create environment
    python -m venv .venv
    
    # Activate (Windows)
    .venv\Scripts\activate
    
    # Activate (Linux/Mac)
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

### 3. Configuration

1.  Open `config.yaml`.
2.  Update `raw_data_folder` to point to your input directory.
3.  Update `processed_folder` to point to the desired flat relational output tree.

### 4. Running the Tool

To run the main processing pipeline:

    python -m src.main


📂 Project Structure
--------------------

    vesperflow/
    ├── config.yaml              # Global settings and paths
    ├── requirements.txt         # Python dependencies
    ├── build_app.py             # PyInstaller script for standalone builds
    ├── external_tools/          # Place GeoTag tools here (Ignored by Git)
    │   └── CG/
    │       ├── GeoTag/          # GeoTag.exe and its sidecar files
    │       └── GeoTagEngine/    # GeoTagEngine.exe and its DLLs
    ├── data_input/              # Input .BIN files (Ignored by Git)
    ├── data_output/             # Final database-ready flat outputs
    └── src/
        ├── main.py              # Pipeline orchestrator
        ├── core/                # Core Application Logic
        │   ├── file_scanner.py  # Crawls raw data & flattens session chunks
        │   ├── export_manager.py# Handles deterministic I/O naming
        │   ├── run_reporter.py  # Generates true yield statistics
        │   ├── binary_decoder.py# Centralized 128-byte/16-byte metadata math
        │   ├── logger.py        # Smart console truncation logging
        │   └── constants.py     # Versioning & magic numbers
        ├── parsers/             # Native Python Vectorized Decoders
        │   ├── imu_parser.py    # Decodes IMU & applies NaN masking
        │   ├── audio_parser.py  # Decodes PCM Audio & strictly writes RIFF
        │   └── gps_parser.py    # Decodes GPS Binary & performs I/Q swap
        ├── tools/               # Standalone diagnostic scripts
        │   ├── imu_inspector.py # IMU binary format inspector & debugger
        │   └── audio_inspector.py # Audio signal integrity checker
        └── wrappers/            # External tool wrappers
            └── geotag_wrapper.py# Wrapper for GeoTag.exe


Contributions
-------------------------- 
We follow Conventional Commits.
<details>
<summary><strong> Please format commit messages as follows: </strong></summary>

- `Feat`: Add native Audio parser
- `Fix`: Resolve 64KB block clicking noise
- `Docs`: Update tools usage
- `Refactor`: Optimize file crawler

**Important:** Do not commit raw data files (.BIN, .DAT).
</details>
   
License & Legal Notice
----------------------------
**VesperFlow** is an unofficial, independent, open-source tool. This project is licensed under the **GNU General Public License v3.0**, see the [LICENSE](LICENSE) for details.

<details>
<summary><strong> Click to Read Legal Notice </strong></summary>
   
1.  **Non-Affiliation:** This project is not in any way officially connected with A.S.D. (Alexander Schwartz Developments), or any of its subsidiaries. The official A.S.D. website can be found at asd-tech.com.
2.  **Trademarks:** The names Vesper, VesperTag, and VesperApp are registered trademarks of A.S.D. Use of these names within this project is strictly for nominative purposes to identify the specific hardware data formats this tool is designed to process.
3.  **Independent Implementation:** While public documentation and legacy references were consulted to understand data structures, this Software was built from scratch. The processing architecture was independently developed using modern data science libraries to ensure high performance and data integrity. No source code was translated or ported from the original manufacturer's software.
</details>

[dpz-badge]: https://img.shields.io/badge/Developed_at-DPZ-009941?logo=data:image/svg+xml;base64,PHN2ZyB2ZXJzaW9uPSIxLjIiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgdmlld0JveD0iMCAwIDQ2MiA0NjAiIHdpZHRoPSI0NjIiIGhlaWdodD0iNDYwIj4KCTxzdHlsZT4KCQkuczAgeyBmaWxsOiAjMDA5OTQxIH0gCgk8L3N0eWxlPgoJPGcgaWQ9IkJhY2tncm91bmQiPgoJCTxwYXRoIGlkPSJQYXRoIDEiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xhc3M9InMwIiBkPSJtMjM2IDAuNDZjMjUuMjIgMC4zMSAyNy4yMiAwLjQ3IDQxLjUgMy40NiA4LjI1IDEuNzMgMjEuMDcgNS4xOSAyOC41IDcuNjggNy40MyAyLjQ5IDE5LjggNy42IDI3LjUgMTEuMzUgNy43IDMuNzUgMTguNzMgOS45NSAyNC41IDEzLjc3IDUuNzcgMy44MiAxNC4xIDkuOCAxOC41IDEzLjI4IDQuNCAzLjQ4IDEzLjE5IDExLjU0IDE5LjUzIDE3LjkxIDYuMzQgNi4zOCAxNS4zOSAxNi43NiAyMC4xIDIzLjA5IDQuNzIgNi4zMyAxMC44NyAxNS4zMyAxMy42NyAyMCAyLjggNC42OCA3LjgxIDE0LjI0IDExLjE0IDIxLjI1IDMuMzMgNy4wMSA2LjA2IDEzLjY1IDYuMDYgMTQuNzUgMCAxLjEgMC40IDIuMTEgMC44OCAyLjI1IDAuNDkgMC4xNCAyLjQ3IDUuMiA0LjQxIDExLjI1IDEuOTQgNi4wNSA0LjQzIDE1LjI4IDUuNTMgMjAuNSAxLjEgNS4yMiAyLjU0IDEzLjU1IDMuMiAxOC41cTEuMiA5IDAuNzEgMzYuNWMtMC40NCAyNC44LTAuNzQgMjguNy0zLjExIDM5Ljc1LTEuNDQgNi43NC00LjI1IDE3LjMxLTYuMjQgMjMuNS0xLjk5IDYuMTktNC4wMSAxMS4zNi00LjUgMTEuNS0wLjQ4IDAuMTQtMC44OCAxLjE1LTAuODggMi4yNSAwIDEuMS0xLjgyIDYuMTYtNC4wNSAxMS4yNS0yLjIzIDUuMDktNi4zNSAxMy4zLTkuMTYgMTguMjUtMi44MSA0Ljk1LTcuOSAxMy4wNS0xMS4zMSAxOC0zLjQxIDQuOTUtOS43OSAxMy4yNy0xNC4xOCAxOC41LTQuMzkgNS4yMy0xMS43NiAxMy4wNS0xNi4zOSAxNy4zOS00LjYzIDQuMzQtMTIuMDEgMTAuNzgtMTYuNDEgMTQuMzEtNC40IDMuNTMtMTMuMTggOS43MS0xOS41IDEzLjczLTYuMzIgNC4wMi0xNS4yMSA5LjE2LTE5Ljc1IDExLjQ0LTQuNTQgMi4yNy0xMi42NCA1Ljg4LTE4IDguMDItNS4zNiAyLjEzLTE0LjI1IDUuMjMtMTkuNzUgNi44OC01LjUgMS42NC0xNC43MyAzLjk0LTIwLjUgNS4xLTUuNzcgMS4xNS0xNC41NSAyLjU1LTE5LjUgMy4xMS00Ljk1IDAuNTUtMTIuOTQgMS4wMS0yNi41IDEuMDJ2LTExNWgxMy43NWM3LjU2IDAgMTcuNTctMC40NyAyMi4yNS0xLjA1IDQuNjgtMC41OCAxMS44Ny0xLjc0IDE2LTIuNTcgNC4xMi0wLjgzIDEyLjIzLTMuMDggMTgtNSA1Ljc3LTEuOTIgMTMuODctNS4xIDE4LTcuMDYgNC4xMi0xLjk3IDExLjEtNS45MiAxNS41LTguNzggNC40LTIuODUgMTEuODItOC4zOSAxNi41LTEyLjMxIDQuNjgtMy45MiAxMC43NS05LjY5IDEzLjUtMTIuODEgMi43NS0zLjEyIDcuNTctOS4xMSAxMC43MS0xMy4zIDMuMTMtNC4xOSA3Ljk0LTExLjg5IDEwLjY4LTE3LjEyIDIuNzUtNS4yMyA2LjU3LTE0LjIzIDguNTEtMjAgMS45My01Ljc3IDQuMjYtMTQuNzggNS4xNy0yMCAxLjI3LTcuMzUgMS41My0xMy4yNCAxLjEyLTI2LTAuMzEtOS41Ny0xLjIyLTE5LjY1LTIuMTgtMjQtMC45MS00LjEyLTMuMDItMTEuNTUtNC43LTE2LjUtMS42OC00Ljk1LTUuMDktMTMuMDUtNy41Ny0xOC0yLjQ4LTQuOTUtNi42My0xMi4xNS05LjIzLTE2LTIuNTktMy44NS03LjU5LTEwLjM1LTExLjExLTE0LjQ0LTMuNTItNC4wOC05Ljc4LTEwLjMxLTEzLjktMTMuODItNC4xMy0zLjUxLTkuOTgtOC4wMy0xMy0xMC4wMy0zLjAyLTItOS41NS01Ljc1LTE0LjUtOC4zNC00Ljk1LTIuNTktMTMuNzMtNi4zNS0xOS41LTguMzUtNS43Ny0yLTE1LjU2LTQuNTEtMjEuNzUtNS41OC03LjA3LTEuMjItMTUuOS0xLjk0LTIzLjc1LTEuOTQtNy45MyAwLTE2LjYxIDAuNzItMjMuNzUgMS45Ni02LjE5IDEuMDgtMTUuNTMgMy4zMS0yMC43NSA0Ljk1LTUuMjIgMS42NS0xNC40NSA1LjUxLTIwLjUgOC42LTYuMDUgMy4wOC0xNC41MyA4LjE2LTE4Ljg1IDExLjI5LTQuMzIgMy4xNC0xMS4yNiA4Ljg1LTE1LjQyIDEyLjctNC4xNiAzLjg1LTEwLjQ3IDEwLjYtMTQuMDIgMTUtMy41NCA0LjQtOC44MyAxMi4wNS0xMS43NSAxNy0yLjkyIDQuOTUtNi45NCAxMi44Mi04Ljk0IDE3LjUtMiA0LjY4LTQuNzQgMTIuMzItNi4wNyAxNy0xLjM0IDQuNjgtMy4xMSAxMy0zLjk0IDE4LjUtMC44NiA1LjczLTEuNSAxNi45NC0xLjUxIDQyLjVsLTExNC41LTAuNS0wLjMtNmMtMC4xNy0zLjMgMC40My0xMi4zIDEuMzItMjAgMC45LTcuOCAzLjMyLTIwLjY0IDUuNDYtMjkgMi4xMi04LjI1IDUuNDktMTkuMjggNy41LTI0LjUgMi01LjIyIDYuMDQtMTQuNDUgOC45OS0yMC41IDIuOTQtNi4wNSA3LjktMTUuMDUgMTEuMDEtMjAgMy4xMS00Ljk1IDguODktMTMuMjggMTIuODMtMTguNSAzLjk0LTUuMjIgMTMuMDItMTUuMzUgMjAuMTgtMjIuNSA3LjE2LTcuMTUgMTcuNTEtMTYuNCAyMy4wMS0yMC41NCA1LjUtNC4xNSAxNS40LTEwLjc3IDIyLTE0LjcyIDYuNi0zLjk1IDE4LjE5LTkuOSAyNS43NS0xMy4yMSA3LjU2LTMuMzIgMTguODEtNy41NyAyNS05LjQ1IDYuMTktMS44OCAxNS45Ny00LjM4IDIxLjc1LTUuNTYgNS43OC0xLjE4IDEzLjY1LTIuNTMgMTcuNS0zLjAxIDMuODUtMC40NyAxOC45My0wLjcyIDMzLjUtMC41NXptLTExMyAyMzEuNTRoMTAydjEwMmgtMTAyeiIvPgoJPC9nPgo8L3N2Zz4=
