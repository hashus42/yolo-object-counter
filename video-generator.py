# import cv2
# import numpy as np
# from PIL import Image
# import os
# import random
#
# # Parameters
# frame_width = 800
# frame_height = 600
# fps = 100
# duration = 60  # in seconds
# num_images = 5  # number of images on the conveyor belt
#
# # Control variable: if 1, save the video; otherwise, do not
# save_video = 1  # Set this to 0 if you don't want to save the video
#
# # Directory containing images
# image_directory = 'images'
#
# # Function to load all image files from the directory
# def load_images_from_directory(directory):
#     # Supported image extensions
#     supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
#     # List all files in the directory and filter by supported extensions
#     image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in supported_extensions]
#     return image_files
#
# # Load images from the directory
# image_files = load_images_from_directory(image_directory)
#
# # Ensure there are enough images for the conveyor belt
# if len(image_files) < num_images:
#     raise ValueError(f"Not enough images in the directory. Found {len(image_files)}, but need {num_images}.")
#
# # Load and resize the images
# images = [Image.open(img_file).resize((150, 150)) for img_file in image_files[:num_images]]  # Resize images to a smaller size
#
# # Convert images to OpenCV format (BGR)
# images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]
#
# # Create a video writer if save_video is 1
# if save_video == 1:
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     video_file = "output_conveyor_belt.mp4"
#     out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))
#
# # Number of frames in the video
# total_frames = fps * duration
#
# # Initialize image positions (start below the frame)
# positions = [frame_height + i * (frame_height // num_images) for i in range(num_images)]
#
# # Create the video frames
# # Create the video frames
#
# # Create the video frames
# for frame_idx in range(total_frames):
#     # Create a white background
#     frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
#
#     # Update image positions
#     for i in range(num_images):
#         img_height, img_width, _ = images_cv2[i].shape
#         y_pos = positions[i] - (frame_idx % (frame_height + img_height))  # Allow images to overflow
#
#         # Ensure the image is within frame boundaries for rendering
#         if y_pos < frame_height:
#             # Clip the part of the image that is within the frame's height boundaries
#             y_start = max(y_pos, 0)
#             y_end = min(y_pos + img_height, frame_height)
#
#             # Adjust the part of the image being drawn, so it doesn't overflow
#             img_y_start = 0 if y_start == y_pos else y_start - y_pos
#             img_y_end = img_height if y_end == y_pos + img_height else y_end - y_pos
#
#             # x-axis is centered
#             x_pos = (frame_width // 2) - (img_width // 2)
#             frame[y_start:y_end, x_pos:x_pos + img_width] = images_cv2[i][img_y_start:img_y_end]
#
#
#     # If save_video is 1, write the frame to the video
#     if save_video == 1:
#         out.write(frame)
#
# # Release the video writer if save_video is 1
# if save_video == 1:
#     out.release()
#     print("Video created successfully.")
#
#
# # Function to display the video
# def play_video(video_file):
#     cap = cv2.VideoCapture(video_file)
#
#     if not cap.isOpened():
#         print("Error: Couldn't open video file.")
#         return
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Display the video frame
#         cv2.imshow('Generated Video', frame)
#
#         # Press 'q' to exit the video display
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#
#     # Release the video capture and close any OpenCV windows
#     cap.release()
#     cv2.destroyAllWindows()
#
# # After generating the video, play it
# if save_video == 1:
#     play_video(video_file)
# else:
#     print("Video not saved, save_video is set to 0.")

import cv2
import numpy as np
from PIL import Image
import os
import random

# Parameters
frame_width = 800
frame_height = 600
fps = 30
num_images = 14  # number of images on the conveyor belt
image_height = 150  # Height of each image

# Control variable: if 1, save the video; otherwise, do not
save_video = 1  # Set this to 0 if you don't want to save the video

# Directory containing images
image_directory = 'images'

# Function to load all image files from the directory
def load_images_from_directory(directory):
    # Supported image extensions
    supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    # List all files in the directory and filter by supported extensions
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in supported_extensions]
    return image_files

# Load images from the directory
image_files = load_images_from_directory(image_directory)

# Ensure there are enough images for the conveyor belt
if len(image_files) < num_images:
    raise ValueError(f"Not enough images in the directory. Found {len(image_files)}, but need {num_images}.")

# Load and resize the images
images = [Image.open(img_file).resize((image_height, image_height)) for img_file in image_files[:num_images]]  # Resize images to 150x150 pixels

# Convert images to OpenCV format (BGR)
images_cv2 = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

# Calculate how long it takes for the last image to completely leave the frame
# The farthest image starts from below the frame and moves its entire height out of the frame.
# Calculate how long it takes for the last image to completely leave the frame
gap_between_images = 50
last_image_start_position = frame_height + (num_images - 1) * (image_height + gap_between_images)
total_frames = int((last_image_start_position + image_height) / (image_height / fps))  # Ensure enough frames for the last image to leave

# Create a video writer if save_video is 1
if save_video == 1:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_file = "output_conveyor_belt.mp4"
    out = cv2.VideoWriter(video_file, fourcc, fps, (frame_width, frame_height))

# Initialize image positions (start below the frame)
positions = [frame_height + i * (image_height + gap_between_images) for i in range(num_images)]
x_pos = [(frame_width // 3) - 40 + (i * 40) for i in range(1, 4, 2)]

# Create the video frames
for frame_idx in range(total_frames):
    # Create a frame with a white, black, and white background
    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:, :frame_width//3, :] = (128, 128, 128)
    frame[:, 2*frame_width//3:, :] = (128, 128, 128)

    # Update image positions
    for i in range(num_images):
        img_height, img_width, _ = images_cv2[i].shape
        y_pos = positions[i] - (frame_idx * image_height // fps)

        # Ensure the image is within frame boundaries for rendering
        if y_pos + img_height > 0:
            # Clip the part of the image that is within the frame's height boundaries
            y_start = max(y_pos, 0)
            y_end = min(y_pos + img_height, frame_height)

            # Adjust the part of the image being drawn, so it doesn't overflow
            img_y_start = max(0, y_start - y_pos)
            img_y_end = img_y_start + (y_end - y_start)  # Match the height of both slices

            # Only place the image if there's a valid slice to draw
            if y_end > y_start and img_y_end > img_y_start:
                # Center the image on the x-axis
                frame[y_start:y_end, x_pos[i % 2]:x_pos[i % 2] + img_width] = images_cv2[i][img_y_start:img_y_end]

    # If save_video is 1, write the frame to the video
    if save_video == 1:
        out.write(frame)

# Release the video writer if save_video is 1
if save_video == 1:
    out.release()
    print("Video created successfully.")


# Function to display the video
def play_video(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("Error: Couldn't open video file.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Display the video frame
        cv2.imshow('Generated Video', frame)

        # Press 'q' to exit the video display
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release the video capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# After generating the video, play it
if save_video == 1:
    play_video(video_file)
else:
    print("Video not saved, save_video is set to 0.")
