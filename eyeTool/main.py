#!/usr/bin/env python3
"""
eyeTool - Camera image loading application using OpenCV
"""

import cv2
import numpy as np


def load_camera_feed():
    """
    Initialize camera and display live feed
    Press 'q' to quit the application
    """
    # Initialize camera (0 is usually the default webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        input("Press Enter to continue...")
        return

    print("Camera feed started. Press 'q' to quit.")
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display the frame
        cv2.imshow('eyeTool - Camera Feed', frame)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed closed.")


def capture_single_image():
    """
    Capture a single image from camera and save it
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        input("Press Enter to continue...")
        return
    
    print("Press SPACE to capture image, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        cv2.imshow('eyeTool - Capture Mode', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key to capture
            filename = 'captured_image.jpg'
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break
        elif key == ord('q'):  # q key to quit
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """
    Main function with menu options
    """
    print("=== eyeTool - Camera Application ===")
    print("1. Live camera feed")
    print("2. Capture single image")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            load_camera_feed()
        elif choice == '2':
            capture_single_image()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        input("\nPress Enter to exit...")
