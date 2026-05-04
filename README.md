# 🌱 Plant Health Monitor (Hybrid)

An AI-powered plant-care system for Raspberry Pi 5 that photographs your plants, analyses their health, and tracks progress over time. Runs in three modes:

- **☁️ Cloud mode** — Claude vision API. Highest quality analysis, best at unfamiliar species. Requires internet + API key.
- **📴 Local mode** — Fully offline on the Pi. Classical computer vision plus an optional TFLite disease classifier. Best as a *trend* tool for catching changes over time, not as a diagnostic tool (more below).
- **🔀 Auto mode** (default) — Tries cloud; seamlessly falls back to local when offline or if the API fails.

The rest of the system — SQLite database, Flask web UI, annotated captures, history charts, care log — is identical across modes. Everything plugs into a common analyzer interface, so readings from any mode are stored and displayed the same way.

## Honest comparison: what each mode can actually do

| Signal | Cloud (Claude) | Local (CV heuristics) | Local (+ TFLite model) |
|---|---|---|---|
| Overall health trend over time | ✅ excellent | ✅ reliable | ✅ reliable |
| Dominant yellowing → overwatering | ✅ | ✅ | ✅ |
| Dominant browning → underwatering | ✅ | ✅ | ✅ |
| Subtle early warning signs | ✅ | ❌ | ⚠️ model-dependent |
| Named diseases (blight, rust, mildew) | ✅ | ❌ | ✅ (14 crop species only) |
| Species identification | ✅ | ❌ | ✅ (PlantVillage crops only) |
| Sun scorch detection | ✅ | ⚠️ prone to false positives on bright backgrounds | ✅ |
| Insufficient light diagnosis | ✅ | ⚠️ unreliable from single photo | ⚠️ same |
| Pest detection (spider mites, etc.) | ✅ | ❌ | ⚠️ model-dependent |
| Care recommendations | ✅ nuanced | ✅ rule-based | ✅ rule-based |
| Works offline | ❌ | ✅ | ✅ |

**If you need real diagnostic quality offline**, train the TFLite model (see below) OR add hardware sensors (soil moisture, lux). Classical CV on its own is best thought of as a "something changed" detector, not a "here's the diagnosis" detector.

## Hardware

- Raspberry Pi 5 running Raspberry Pi OS (Bookworm)
- Official Raspberry Pi Camera Module (v2, v3, or HQ)
- Optional but very helpful: tripod/wall mount for consistent framing

## Installation

```bash
# 1. Copy this directory to your Pi, e.g. ~/plant_monitor
cd ~/plant_monitor

# 2. Create a venv with system site packages (for picamera2)
python3 -m venv --system-site-packages venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install tflite-runtime for disease detection
pip install tflite-runtime

# 5. (Optional) Set your API key for cloud mode
export ANTHROPIC_API_KEY="sk-ant-..."

# 6. Choose default mode (defaults to "auto" — tries cloud, falls back to local)
export PLANT_ANALYZER="auto"   # or "cloud" or "local"

# 7. Initialise the database
python src/cli.py init

# 8. Check what's available
python src/cli.py status
```

Expected `status` output:

```
Analyzer status:
  cloud (Claude API):   ✅ available
  local (offline CV):   ✅ available
  local disease model:  ⚠️  not installed (optional)
```

## Usage

### CLI

```bash
# Add a plant
python src/cli.py add-plant "Kitchen Monstera" --species "Monstera deliciosa"

# Capture using the default mode
python src/cli.py capture 1

# Force a specific mode
python src/cli.py capture 1 --mode local
python src/cli.py capture 1 --mode cloud

# See history — the Analyzer column shows which backend produced each reading
python src/cli.py history 1

# Log a care event
python src/cli.py care 1 watered --notes "Gave about 200ml"
```

### Web UI

```bash
python src/web_app.py
```

Browse to `http://<pi-ip>:5000`. The header shows the current mode and a cloud/offline indicator. On each plant's page, a dropdown next to the Capture button lets you override the analyzer per-capture. Every reading is tagged with the analyzer that produced it.

### Scheduler

```bash
python src/scheduler.py
```

Runs a daily capture of all plants at 09:00 (edit `CAPTURE_TIME` in `scheduler.py`). In auto mode, uses cloud when online, falls back to local when offline.

### Systemd services

```bash
sudo cp plant-monitor-web.service /etc/systemd/system/
sudo cp plant-monitor-scheduler.service /etc/systemd/system/
# Edit the files to set your paths and ANTHROPIC_API_KEY if using cloud
sudo systemctl daemon-reload
sudo systemctl enable --now plant-monitor-web plant-monitor-scheduler
```

## How the local analyzer works — and its limits

Three layers combined:

**Layer 1 — Classical computer vision (always runs):**
- HSV colour segmentation isolates plant pixels from background
- Morphological closing builds a "plant silhouette" that includes leaf interiors even where discoloured
- Measures ratios within the silhouette: healthy green, yellow (chlorosis), brown (necrosis), bleached (scorch)
- Droop score from the vertical position of the plant's centroid
- Scene brightness/saturation used as weak light proxy

**Layer 2 — Optional TFLite disease classifier:**
- MobileNetV2 trained on PlantVillage (38 classes, 14 crop species)
- Runs in ~200ms on the Pi 5 CPU via `tflite-runtime`
- Only triggers if `models/plant_disease.tflite` exists — see below

**Layer 3 — Rule-based synthesis:**
- Combines signals into the same JSON schema Claude produces
- Thresholds in `src/analyzers/local.py` `_synthesise()` are the main knobs
- Confidence scores are deliberately lower than cloud mode when signals are ambiguous

### Known limitations — read this before trusting local mode

**The thresholds ship conservative.** They're tuned not to false-positive — a healthy plant with a bright background (window, wall) won't get flagged as scorched, for example. But this means small real problems may slip through. Classical CV cannot reliably distinguish "3% of the plant is mildly yellow because of the angle of the light" from "3% of the plant is genuinely yellowing from overwatering." The cloud analyzer can read that nuance. The local one cannot.

**Use local mode primarily for trends.** A plant scoring 85 consistently, then dropping to 60 over a week, is a real signal regardless of the absolute thresholds. Compare readings across days, not across plants.

**Single-photo light diagnosis is fundamentally hard.** Adding a £3 BH1750 lux sensor is a much better answer than tuning heuristics. "Insufficient light" and "excessive light" from a single photo are guesses; confidence scores reflect this.

**The calibration script is your friend.** See below.

### Calibrating for your setup

```bash
python calibrate.py photo.jpg
```

This prints every CV signal the analyzer extracts plus the final verdict. Take photos of plants you know are healthy and plants you know have problems, run them through calibrate.py, and adjust thresholds in `src/analyzers/local.py` (`_synthesise()`) until the outputs match your judgment.

Add `--save-masks out/` to dump PNG visualisations of the plant mask and silhouette — useful for spotting segmentation issues.

## Adding a TFLite disease model (optional)

There's no single canonical pre-trained PlantVillage TFLite model available for direct download that I can recommend. Your options:

1. **Train it yourself** on Google Colab using [obeshor/Plant-Diseases-Detector](https://github.com/obeshor/Plant-Diseases-Detector) — takes about an hour on a free Colab GPU, outputs a `.tflite` file. Drop it at `models/plant_disease.tflite`.
2. **Find a pre-trained one** on [Kaggle](https://www.kaggle.com/search?q=plantvillage+tflite). Verify the class labels match the `PLANTVILLAGE_CLASSES` list in `src/analyzers/local.py`.
3. **Skip it.** The CV-only analyzer still works.

See `src/download_model.py` for more detail.

## Architecture

```
plant_monitor/
├── src/
│   ├── plant_monitor.py       # Core: camera, orchestration, DB, annotation
│   ├── web_app.py             # Flask UI
│   ├── scheduler.py           # Daily auto-capture
│   ├── cli.py                 # Command-line tool
│   ├── download_model.py      # TFLite model setup guide
│   └── analyzers/
│       ├── base.py            # Analyzer interface + shared schema
│       ├── cloud.py           # Claude vision API + connectivity check
│       ├── local.py           # Offline CV + optional TFLite
│       └── hybrid.py          # Cloud-first with fallback; factory
├── templates/                 # Jinja templates
├── captures/plant_<id>/       # Photos per plant (original + annotated)
├── models/plant_disease.tflite  # Optional; place here if you have one
├── db/plants.db               # SQLite (includes analyzer column)
├── calibrate.py               # Threshold-tuning tool for local analyzer
├── smoke_test.py              # Schema validation tests
└── requirements.txt
```

Each reading stores which analyzer produced it, so you can filter history, spot drift when switching modes, or compare cloud vs local readings on the same plant by capturing both.

## Extending

- **Soil moisture sensor** (capacitive, via ADS1115 ADC) — huge upgrade for watering accuracy, especially in local mode
- **BH1750 lux sensor** — solves the light-assessment problem properly
- **BME280** — temperature/humidity context stored alongside readings
- **Telegram/Pushover alerts** on poor/critical health
- **Time-lapse videos** from daily captures
- **Dual-mode comparison** — schedule both cloud and local captures daily, log the agreement rate, tune local thresholds from real-world disagreements

To add a new analyzer backend, subclass `analyzers.base.Analyzer`, implement `analyse()` and `is_available()`, and register it in `analyzers/hybrid.py::get_analyzer()`. Nothing else needs to change.

## Troubleshooting

**`cli.py status` says cloud unavailable despite having a key** — the cloud analyzer also checks connectivity to `api.anthropic.com:443`. If your Pi can't reach the internet, it'll report unavailable regardless of the key.

**Local analyzer reports "excellent" on a clearly unhealthy plant** — this is the known conservative-threshold issue. Run `calibrate.py` on the photo, note the ratios, and lower the relevant threshold in `_synthesise()`. Expect to do this with real photos, not trust the defaults.

**`picamera2` import error** — create the venv with `--system-site-packages`. Without it you'll get the placeholder green image instead of real captures.

**TFLite model won't load** — check `python src/cli.py status`. Either `tflite-runtime` isn't installed, the `.tflite` file is missing/corrupt, or the model's input/output dtypes aren't what the code expects (float32 or uint8 quantized are both handled).

## License

MIT.
