# eyeTool

A Python application for camera image loading and processing using OpenCV.

## Features

- Live camera feed display
- Single image capture
- Simple command-line interface

## Installation

1. Navigate to the project directory:
   ```bash
   cd eyeTool
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python main.py
```

### Menu Options

1. **Live camera feed** - Displays real-time camera feed. Press 'q' to quit.
2. **Capture single image** - Capture and save an image. Press SPACE to capture, 'q' to quit.
3. **Exit** - Close the application.

## Requirements

- Python 3.7+
- OpenCV 4.9.0.80
- NumPy 1.26.4

## Project Structure

```
eyeTool/
|-- main.py              # Main application script
|-- requirements.txt      # Python dependencies
|-- README.md            # Project documentation
```
