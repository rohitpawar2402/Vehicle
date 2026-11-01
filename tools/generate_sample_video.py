"""Generates a synthetic CCTV traffic video for testing the violation pipeline without external downloads."""

import cv2
import numpy as np


def create_traffic_video(output_path: str = "data/sample_traffic.mp4", num_frames: int = 60):
    w, h = 1280, 720
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Road background
    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 60  # Asphalt gray

        # Road lane lines
        cv2.line(frame, (400, 0), (300, 720), (255, 255, 255), 3)
        cv2.line(frame, (880, 0), (980, 720), (255, 255, 255), 3)
        cv2.line(frame, (640, 0), (640, 720), (255, 255, 0), 2)  # Center divider

        # Moving car body (moving from top y=100 down to y=650)
        car_y = int(100 + (i / num_frames) * 550)
        car_x = 480
        car_w = 160
        car_h = 100

        # Draw vehicle body (blue car)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (180, 50, 30), -1)
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (255, 255, 255), 2)
        # Windshield
        cv2.rectangle(frame, (car_x + 20, car_y + 15), (car_x + car_w - 20, car_y + 45), (200, 200, 200), -1)
        # Headlights
        cv2.circle(frame, (car_x + 25, car_y + car_h - 10), 10, (0, 255, 255), -1)
        cv2.circle(frame, (car_x + car_w - 25, car_y + car_h - 10), 10, (0, 255, 255), -1)

        writer.write(frame)

    writer.release()
    print(f"[OK] Generated {num_frames} frames sample traffic video at: {output_path}")


if __name__ == "__main__":
    create_traffic_video()

