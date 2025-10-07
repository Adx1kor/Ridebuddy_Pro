# 🚀 RideBuddy Pro v2.1.0 - Quick Reference Card

## Essential Files (DO NOT DELETE)
```
✅ ridebuddy_optimized_gui.py    # Main application
✅ vehicle_launcher.py           # Vehicle deployment  
✅ camera_diagnostics.py         # Hardware testing
✅ requirements.txt              # Dependencies
✅ ridebuddy_config.ini         # App configuration
✅ vehicle_config.json          # Vehicle settings
✅ ridebuddy_data.db            # Application database
✅ logs/ (directory)            # Application logs
✅ src/ (directory)             # Source modules
✅ configs/ (directory)         # Model configs
```

## Quick Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Test camera hardware  
python camera_diagnostics.py

# Launch desktop application
python ridebuddy_optimized_gui.py

# Launch vehicle mode
python vehicle_launcher.py
```

## System Requirements
- **Python**: 3.8 - 3.11
- **RAM**: 4GB minimum (8GB for vehicle mode)
- **Camera**: USB webcam (720p+)
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

## Performance Targets
- **Inference**: <50ms per frame
- **Memory**: <512MB (<384MB vehicle mode)
- **Accuracy**: 70-98% across all detection types
- **Frame Rate**: 25+ FPS

## Troubleshooting
```bash
# Camera issues
python camera_diagnostics.py

# Check logs
cat logs/ridebuddy_*.log

# Memory issues - use vehicle mode
python vehicle_launcher.py
```

## Vehicle Mode Environment
```bash
RIDEBUDDY_VEHICLE_MODE=1
RIDEBUDDY_FLEET_MODE=1
```

**📚 For detailed setup:** See `RIDEBUDDY_FINAL_SETUP_GUIDE.md`