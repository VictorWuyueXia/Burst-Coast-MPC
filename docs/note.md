codebase: https://github.com/raprakashvi/sonopet/tree/robot

D405 capture and stitching:gui/realsense.py
command(sonopet env): python3 gui/realsense.py --robot-ip 172.16.0.2

robot motion: datacollection_franky/run_pipeline.py
command(sonopet env): python3 datacollection_franky/run_pipeline.py --robot-ip 172.16.0.2 --pcd-path /pointclouds/stitched/<**.pcd> --pcd-frame fr3_link0


D405 RGB streaming, microphone, and data saving: recorder/run.sh
command: run.sh