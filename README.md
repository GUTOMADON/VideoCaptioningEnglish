## Requirements
- Python 3.8 or higher
- Required packages: torch, numpy, and any other dependencies listed in the code

## Setup
1. Open a terminal in the project folder.
2. Install the required packages:
   ```bash
   pip install torch numpy
   ```
   Add other packages if needed.

## How to Run
1. Place your video files in the appropriate folder if needed.
1. Put your video file (e.g., `video.mp4`) in the `VideoAnomaly` folder or as required by the script.
2. Run the main script:
   ```bash
   python VideoAnomaly/VideoAnomaly.py
   ```
3. The script will process the videos and generate a report in the output folder.
## Example Terminal Output
You will see progress messages like:
```
Processing video.mp4...
Anomaly detected at frame 120
Report saved to crash_output_real/report.json
```

## Output
- Reports and results will be saved in the `crash_output_real` or `VideoAnomaly` folders.

## Troubleshooting
- If you get errors about missing packages, install them using pip.
- Make sure your Python version matches the requirements.

## Contact
If you have questions or need help, feel free to reach out.
