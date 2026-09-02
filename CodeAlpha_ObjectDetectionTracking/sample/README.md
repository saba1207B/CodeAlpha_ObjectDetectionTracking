# Sample Videos Directory

Place any sample test videos in this directory to run multi-object tracking.

### Recommended Test Video Types:
1. **Pedestrian / Street Walkway:** Videos showing multiple people walking across the frame to test ID persistence and occlusion recovery.
2. **Road Traffic / Vehicles:** Videos showing cars, buses, and motorcycles moving along lanes to test speed and multi-class tracking.
3. **Sports / Indoor Actions:** Fast-moving objects (players, basketballs) to evaluate Kalman filter tracking performance.

### Running Tracking on Sample Videos:
```bash
# Run tracking on a sample video file
python app.py --source sample/traffic.mp4

# Run with custom confidence and trajectory trails
python app.py --source sample/pedestrians.mp4 --conf 0.40 --save output/pedestrians_tracked.mp4
```

*Note: Large sample video files should not be committed to GitHub. They are ignored by `.gitignore`.*
