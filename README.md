# RoadSight AI

AI-powered pothole detection and road maintenance monitoring.

## Features
- YOLOv8n-Seg pothole detection
- Image and video analysis
- Severity classification
- Experimental GeoCLIP location estimation
- Interactive pothole map
- Estimated road repair budget

## Run locally

pip install -r requirements.txt
streamlit run app.py

## Important
GeoCLIP estimates are not verified capture GPS.
Severity is an experimental mask-area proxy.
Repair costs are planning assumptions, not official quotations.
