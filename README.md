# 8ball-detection

This project provides computer vision tools for detecting pool (billiard) balls, the cue ball, and simulating ball trajectories and reflections on a pool table from an image. It uses OpenCV and NumPy for image processing and geometric calculations.

## Features
- **Table Detection:** Identifies the inner blue playing area of a pool table in an image.
- **Ball Detection:** Detects all balls (including the white cue ball) within the table area using color segmentation and Hough Circle Transform.
- **Cue Ball & Cue Stick Detection:** Locates the white ball and estimates the direction of the cue stick.
- **Trajectory Simulation:** Simulates the path of the cue ball, including wall reflections and collisions with other balls.
- **Visualization:** Draws detected balls, cue ball, cue stick, and predicted trajectories on the image for visualization.

## File Overview
- `detect.py` — Main script for detection and simulation. Contains all core logic for image processing, detection, and visualization.
- `detect.txt` — (Reference/alternate) Contains similar or extended detection and simulation logic, possibly with more comments and constants.
- `1.png`, `2.png`, `3.png` — Example images of pool tables for testing.
- `predicted_trajectory.png` — Example output image showing detected balls and predicted trajectory.
- `README.md` — Project documentation (this file).

## Requirements
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy

Install dependencies with:
```sh
pip install opencv-python numpy
```

## Usage
1. Place a pool table image (e.g., `1.png`) in the project directory.
2. Run the detection script:
   ```sh
   python detect.py
   ```
3. The script will display the detected balls, cue ball, and predicted trajectory. Adjust parameters in `detect.py` for different images or detection sensitivity.

## Customization
- **Detection Parameters:** You can tune HSV color ranges, Hough Circle parameters, and simulation constants in `detect.py` or `detect.txt` for different lighting or table conditions.
- **Image Input:** Change the `image_path` variable in `detect.py` to use a different input image.

## License
This project is for educational and research purposes. See source files for details.
 
