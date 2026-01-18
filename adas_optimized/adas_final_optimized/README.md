# ADAS v5.1 Custom - Clean 1080p Build

## 8-10 FPS at 1080p (Lean & Optimized)

### Files Removed
- bumper_detector.py (Moondream2 will handle)
- speed_bump_detector.py (Moondream2 will handle)
- License plate code (unused)

### Quick Start
```bash
tar -xzf adas_v5_custom_1080p.tar.gz
cd adas_v5_custom_1080p
bash setup.sh
python3 main.py video.mp4 --bisenet --output result.mp4
```

Expected: 8-10 FPS at 1080p
Package: 22KB (clean build)
