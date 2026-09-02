# Output Videos Directory

This directory stores rendered video files containing detected bounding boxes, tracking IDs, motion trajectory trails, and the statistics HUD overlay.

### Generating Saved Videos:
```bash
# Save output from video tracking:
python app.py --source sample/traffic.mp4 --save output/traffic_tracked.mp4

# Save output from live webcam feed:
python app.py --source 0 --save output/webcam_tracked.mp4
```

*Note: Generated `.mp4` and `.avi` files are automatically ignored by `.gitignore` to keep the repository lightweight.*
