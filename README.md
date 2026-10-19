🐒 ⚙️ VesperFlow
=====================
![Version](https://img.shields.io/github/v/release/dasilva98/vesperflow)
[![DPZ Lab][dpz-badge]](https://www.dpz.eu/en/about-us)
![License](https://img.shields.io/badge/License-GPLv3-blue)

[Tool Demo Video](https://youtu.be/wNGPoCBlQts)

A headless, scriptable CLI automation pipeline for decoding and processing multi-modal time-series data from bio-logging tags. 

Designed specifically for machine learning and database ingestion workflows, this tool provides a "One-Click" solution to convert raw binary dumps into analysis-ready formats. By bypassing intermediate disk writes in favor of vectorized in-memory array operations, it natively handles proprietary binary decoding, true-time synchronization, software-defined radio (SDR) satellite positioning, and hardware artifact mitigation.

> **DISCLAIMER** This is an unofficial, independent research tool developed by students at the University of Göttingen for the German Primate Center (DPZ). It is **not** affiliated with A.S.D. (Alexander Schwartz Developments). All product names are property of their respective owners and are used here solely for identification and compatibility purposes.

Key Features
-----------------------
The native parsers utilize a standardized **128-byte File Header** and **16-byte Block Header** architecture to unpack payloads, while automatically mitigating known hardware quirks:

Key Features
-----------------------
Vesper tags write raw data using a **128-byte File Header** and **16-byte Block Header** layout with several undocumented hardware quirks. VesperFlow unpacks these streams in memory and cleans up the artifacts before writing analysis-ready files:

### 1. Sensor Processing & Hardware Fixes

#### IMU (Accelerometer, Gyroscope & Magnetometer)
- Reconstructs per-packet timestamps directly from the hardware's 1024-tick sub-second counter using vectorized NumPy operations.
- Masks the reproducible `-8.75` mdps gyroscope glitch at file boundaries with `NaN` so hardware file-split spikes don't corrupt downstream ML classifiers.
- Reads the header bitmask to automatically include **Temperature (°C)** and **Barometric Pressure (hPa)** columns when extended sensors are enabled.

#### Audio (Microphone)
- Strips the 16-byte sync markers injected at every 64 KB flash page boundary, keeping 16-bit PCM sample alignment intact across long recordings.
- Writes standard-compliant 48 kHz `.wav` RIFF headers, fixing the 16-byte chunk truncation bug in proprietary extractors that causes Python audio libraries to crash.
- Mutes the ~17 ms (`0x8000`) DC-offset pop generated during microphone power-on.

#### GPS (Raw Snapshots & Native SDR Solver)
- Skips the 1,008-byte warm-up padding in `0G.BIN`, applies the required 32-bit/16-bit I/Q word swaps, and logs sub-second header timestamps (`subseconds.csv`) before unpacking the 1.024 MHz 1-bit complex I/Q stream.
- Screens snapshots with a fast 80 ms noise gate (skipping indoor or underground recordings in milliseconds) before running a 480 ms code-Doppler compensated FFT search, 20 ms bit-synchronized coherent integration, and 8x FFT sinc peak interpolation.
- Runs an ephemeris-aided second pass that predicts Doppler and code phase from initial locks to pull weaker canopy satellites out of the noise floor.
- Solves 5-state coarse-time positions (Lat, Lon, Alt, receiver clock bias, and tag timestamp error) using Weighted Least-Squares with WGS-84 ellipsoid height anchoring, Klobuchar L1 ionospheric corrections, subset outlier rejection, and a second-pass session clock model.


### Session Inventory & Diagnostics
`RunReporter` merges 256-file hardware chunks into continuous tag deployments, outputs standardized filenames (`YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS_DEVICEID.ext`), and writes a summary `.txt` report card for each run:
- Calculates true recording duration from sample counts (`N_samples / f_s`) rather than wall-clock start/end times, avoiding skew from scheduled sleep cycles.
- Flags valid headers with empty payloads as **WARNING** (Empty) rather than **FAIL**, separating normal battery exhaustion from real file corruption.
- Exports `Track-sdr-diagnostics.csv` alongside the main GPS track to log locked PRNs, C/N0, pseudorange residuals, and rejection reasons for every snapshot.


### Cross-Platform GPS Modes (`sdr` vs. `geotag`)
- **Native SDR Mode (`gps_solver: "sdr"`):** Decodes GPS snapshots natively on Linux, macOS, and Windows across all CPU cores without external `.exe` dependencies.
- **Automatic Orbit Caching:** Downloads daily NASA/IGS broadcast RINEX v3 navigation files (`BRDC`) into `./data/ephemeris/` on the first run for a deployment date and runs offline thereafter.
- **Regional Search Priors:** Supports continent, country, or field-site presets (`europe`, `germany`, `madagascar`, `affenwald`, etc.) in `config.yaml`, and narrows the search box around the animal after the first solid fix.
- **Legacy GeoTag Mode (`gps_solver: "geotag"`):** Optional Windows wrapper for `GeoTag.exe` / `GeoTagEngine.exe` if you want to cross-check against manufacturer outputs.


Build & Quick Start
-----------------------

### 1. Prerequisites

- **Python 3.12+**

- **Full Cross-Platform Support:** IMU, Audio, and GPS decoding run natively on **Windows, Linux, and macOS**.

- **Internet Connection (First GPS Run Only):** Required once per deployment date to automatically fetch daily IGS/NASA satellite broadcast ephemerides (~300 KB, cached locally in `./data/ephemeris/`).

- **Legacy GeoTag Mode (Optional):** Only required if `gps_solver` is explicitly set to `"geotag"` in `config.yaml` (requires Windows 10/11 and placing `GeoTag.exe` / `GeoTagEngine.exe` in `external_tools/CG/`).


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

1. Open `config.yaml`.
2. Update `raw_data_folder` (e.g., `./data/raw`) and `processed_folder` (e.g., `./data/processed`).
3. Configure the GPS solver settings:
   - `gps_solver: "sdr"` (default native cross-platform solver) or `"geotag"` (legacy Windows wrapper).
   - `gps_region: "europe"` (or `"global"`, `"germany"`, `"panama"`, `"madagascar"`, `"new_zealand"`, etc.) to bound the 
   initial coarse Doppler search.
   - `ephemeris_folder: "./data/ephemeris"` for local RINEX orbit caching.

### 4. Running the Tool

To run the main processing pipeline:

    python -m src.main

To inspect satellite locks, C/N0, Doppler shifts, and WLS pseudorange residuals on a single `.dat` snapshot:

    python -m src.tools.gps_inspector <path/to/snap.YYYY_MM_DD_HH_MM_SS_GC0.dat>


Project Structure
--------------------

    vesperflow/
    ├── config.yaml ---------------> # Global settings, paths, and GPS solver region priors
    ├── requirements.txt ----------> # Python dependencies
    ├── build_app.py --------------> # PyInstaller script for standalone builds
    ├── external_tools/ -----------> # (Optional) Place legacy GeoTag tools here (Ignored by Git)
    │   └── CG/
    │       ├── GeoTag/ -----------> # GeoTag.exe and its sidecar files (Optional)
    │       └── GeoTagEngine/ -----> # GeoTagEngine.exe and its DLLs (Optional)
    ├── data/ ---------------------> # Local data directory (Ignored by Git)
    │   ├── raw/ ------------------> # Input raw session folders (Binary files)
    │   ├── processed/ ------------> # Final database-ready flat outputs (.csv, .wav, reports)
    │   └── ephemeris/ ------------> # Cached daily IGS/NASA RINEX broadcast orbit files
    └── src/
        ├── main.py ---------------> # Pipeline orchestrator & CLI dashboard
        ├── core/ -----------------> # Core Application Logic
        │   ├── file_scanner.py ---> # Crawls raw data & flattens session chunks
        │   ├── export_manager.py -> # Chunked csv/wav exporter & deterministic I/O naming
        │   ├── run_reporter.py ---> # Generates true yield statistics & report cards
        │   ├── binary_decoder.py -> # Centralized 128-byte/16-byte metadata decoder
        │   ├── logger.py ---------> # Tqdm logging & AnimatedTqdm spinner
        │   └── constants.py ------> # Versioning & geographic REGION_PRESETS
        ├── parsers/ --------------> # Native Python Vectorized Decoders
        │   ├── imu_parser.py -----> # Decodes IMU
        │   ├── audio_parser.py ---> # Decodes PCM Audio
        │   └── gps_parser.py -----> # Extracts GPS snapshots, I/Q word swap & sub-second log
        ├── sdr/ ------------------> # Native Cross-Platform GNSS SDR & Navigation Engine
        │   ├── frontend.py -------> # 1.024 MHz 1-bit Complex I/Q unpacker & DC blocker
        │   ├── acquisition.py ----> # PRN generator, 480ms FFT correlator & 20ms bit-sync
        │   ├── ephemeris.py ------> # RINEX v3 auto-downloader, Kepler orbits & Klobuchar iono
        │   └── solver.py ---------> # Region Doppler init, Coarse-Time WLS & Pass-2 clock model
        ├── tools/ ----------------> # Standalone diagnostic scripts
        └── wrappers/ -------------> # External tool wrappers
            └── geotag_wrapper.py -> # [Legacy] Subprocess wrapper for Windows GeoTag.exe


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
