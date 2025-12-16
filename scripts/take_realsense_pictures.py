import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Create output directory
output_dir = "C:\\Users\\Caspe\\Documents\\Semester 9\\EIRT\\real_dataset"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "rgb"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)

print("Press 'c' to capture images, 'q' to quit")

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (for visualization)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally for display
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.imshow('RealSense D415 - Press "space" to capture, "q" to quit', images)
        
        key = cv2.waitKey(1)
        
        # Capture images on 'space' key press
        if key & 0xFF == ord(' '):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save RGB image
            rgb_filename = os.path.join(output_dir, "rgb", f"{timestamp}.png")
            cv2.imwrite(rgb_filename, color_image)
            
            # Save depth image (raw depth data)
            depth_filename = os.path.join(output_dir, "depth", f"{timestamp}.png")
            cv2.imwrite(depth_filename, depth_image)
            
            print(f"Saved: {rgb_filename} images recorded so far: {len(os.listdir(os.path.join(output_dir, 'rgb')))}")
        
        # Quit on 'q' key press
        elif key & 0xFF == ord('q'):
            print("Quitting...")
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()