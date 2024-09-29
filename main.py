import cv2
from ultralytics import YOLO, solutions
import time


def run_video_processing():
    # Load the YOLO model
    model = YOLO("/home/pardusumsu/code/Counting-Sheep/drone-detect.pt")

    # Open the video file
    cap = cv2.VideoCapture("drones-on-conveyor.mp4")
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Define region points for object counting
    # [bottom_left, bottom_right, top_right, top_left]
    region_points = [(20, 400), (w-40, 400), (w-40, 300), (20, 300)]

    # Video writer setup to save the output video
    video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Initialize Object Counter
    counter = solutions.ObjectCounter(
        view_img=False,  # Set to False if you don't want to display the window
        reg_pts=region_points,
        names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    # Initialize variables for FPS calculation
    time_deltas = []
    fps_display = 0

    prev_time = time.time()

    paused = False
    restart = False

    while cap.isOpened():
        current_time = time.time()
        time_elapsed = current_time - prev_time
        prev_time = current_time

        # Calculate FPS based on time elapsed
        if time_elapsed > 0:
            time_deltas.append(time_elapsed)
            # Keep only the last 10 time intervals
            if len(time_deltas) > 10:
                time_deltas.pop(0)
            # Calculate the average time delta and FPS
            avg_time_delta = sum(time_deltas) / len(time_deltas)
            fps_display = 1 / avg_time_delta

        # Read the next frame
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        # Flip the frame horizontally (if needed)
        im0 = cv2.flip(im0, 1)

        # Display FPS on the video
        cv2.putText(im0, f"FPS: {round(fps_display)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (70, 70, 70), 2)

        # Track objects using the YOLO model
        tracks = model.track(im0, persist=True, show=False)

        # Count objects and annotate the frame
        im0 = counter.start_counting(im0, tracks)

        # Write the processed frame to the output video file
        video_writer.write(im0)

        # Display the frame (optional)
        cv2.imshow("Object Counting", im0)
        key = cv2.waitKey(10)
        if key & 0xFF == ord("q"):
            cv2.putText(im0, "Press 'r' to restart or any other key to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Object Counting", im0)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("r"):
                restart = True
            else:
                restart = False
            break
        elif key == ord(" "):
            paused = not paused
            while paused:
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key == ord("q"):
                    break

    # Release the video capture and writer resources
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    return restart


def main():
    while True:
        restart = run_video_processing()
        if not restart:
            break


if __name__ == "__main__":
    main()