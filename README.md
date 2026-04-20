**Plant Health Monitoring System for Raspberry Pi 5**

__Architecture Overview__

The system will combine:
- Picamera2 (native Pi 5 camera library) for image capture
- Claude's Vision API for plant health analysis (excellent for nuanced visual assessment)
- SQLite for lightweight local storage of plant records
- Flask web interface so you can view it from any device on your network
- PIL/Pillow for image annotation


__How it all fits together__

Core flow: plant_monitor.py handles the pipeline — picamera2 captures a high-res JPEG, it's base64-encoded and sent to Claude's vision API with a structured-JSON prompt, the response is parsed, a bounding box and health summary are drawn onto a copy of the image with Pillow, and everything (original path, annotated path, health scores, observations, recommendations, raw JSON) is stored in SQLite.
Three ways to use it: the Flask web UI (web_app.py) for browsing from your phone, the CLI (cli.py) for scripting and quick checks, and the scheduler (scheduler.py) for hands-off daily monitoring. The two systemd unit files let you run the web UI and scheduler as proper background services that start on boot.

Storage layout: each plant gets its own folder under captures/plant_<id>/, containing paired <timestamp>_original.jpg and <timestamp>_annotated.jpg files. The SQLite database at db/plants.db has three tables: plants, readings, and care_events — the care log is key because it lets you see things like "health dropped after I moved it to the bathroom" or "the watering I added on Tuesday helped."

__A few things to note on the Pi__

Create your venv with --system-site-packages — picamera2 is installed system-wide on Raspberry Pi OS and isn't available via pip, so a normal venv won't see it.
The code has a dev fallback — if picamera2 can't be imported, it writes a placeholder green image so you can test the pipeline on non-Pi machines.
Set ANTHROPIC_API_KEY before running, either as an env var or directly in the systemd unit files.
Consistent framing matters a lot — if you can mount the camera in a fixed position pointing at each plant (or use a small rotating rig), your time-series comparisons become much more meaningful.


**ENHANCED: Plant Health Monitor — Hybrid (Cloud + Local Offline)**

Cloud mode — Claude vision API (highest quality)
Local mode — fully offline on the Pi 5
Auto mode — try cloud, fall back to local if no internet

__How the local offline analyzer works__

The Pi 5 has enough compute to run real ML locally, but we need to be realistic — a fine-tuned plant-disease CNN + classical computer vision heuristics gives the best balance of accuracy and speed. 

My approach:
Layer 1 — Classical CV (always fast, always runs):

- Color analysis in HSV space — measures green vs. yellow/brown ratios across the plant to detect chlorosis (overwatering/nutrient issues) or browning (underwatering/scorch)
- Leaf segmentation — isolates plant pixels from background using green-channel thresholding
- Droop/wilt detection — analyzes leaf orientation via edge detection and contour angles
- Brightness/exposure analysis — scene brightness helps infer light conditions when compared over time
- Leaf scorch detection — bright white/bleached patches indicate excessive sunlight

Layer 2 — Deep learning (optional, one-time download):

PlantVillage-trained MobileNetV2 — detects 38 classes of plant diseases (leaf spots, blights, mildews, rusts). Runs via TensorFlow Lite on the Pi 5's CPU in ~200ms per image.

Layer 3 — Rule-based synthesis — combines signals into the same JSON schema the cloud analyzer produces, so the rest of the app doesn't care which one ran.



__What I should have done from the start__
Shipped with confidence scores that honestly reflect uncertainty, not tried to make synthetic tests pass. 

The local analyzer's real value is:

- Reliable detection of widespread color changes (a clearly yellow-dominated plant → yes, that's overwatering/deficiency)
- Consistent measurements over time (score dropping from 85 to 60 over two weeks is meaningful even if the absolute thresholds are off)

A framework to build on with real data

- The local analyzer cannot reliably do single-photo disease diagnosis without a trained model, and my tuning attempts confirmed that. The right answer was what I built anyway — an optional TFLite hook — plus being honest that classical CV alone is a health trend tool, not a diagnostic tool.

__Recommendation__
Rather than burning more turns on synthetic-image tuning, get more value by:

- Taking v1 and using it for a week with real plants via cloud mode
- Dropping those real photos into a calibration script that prints every CV signal
- Tuning thresholds against ground truth observed




**Suggested Additional FUTURE Features**
- Multi-plant tracking with QR codes or position-based IDs — tag each plant location so the system knows which plant it's photographing
- Scheduled captures (cron-based) for consistent daily monitoring
- Environmental sensor integration — a cheap DHT22 or BME280 gives temperature/humidity context
- Soil moisture sensor integration — capacitive sensors via ADC for ground-truth watering data
- Alert notifications via email/Telegram/Pushover when a plant shows distress
- Leaf/pest detection — spots, yellowing, pest damage, fungal signs
- Growth tracking — measure plant size/leaf count over time via bounding boxes
- Time-lapse generation from daily captures
- Comparative analysis — "this plant is doing worse than last week"
- Care recommendations log — track what you did (watered, moved, fertilized) and correlate with health outcomes
- Light level measurement using a BH1750 lux sensor
- Species-specific knowledge — store the species per plant so analysis is tailored
