"""Converts an OpenCV mp4v video to HTML5 browser-compatible H.264 (yuv420p) using PyAV."""

import os
import sys
import av


def reencode_to_h264(input_path: str, output_path: str):
    if not os.path.exists(input_path):
        print(f"[ERROR] Input video not found: {input_path}")
        return False

    print(f"[INFO] Re-encoding '{input_path}' to HTML5-compatible H.264: '{output_path}'...")
    input_container = av.open(input_path)
    in_video = input_container.streams.video[0]

    output_container = av.open(output_path, mode="w")
    out_video = output_container.add_stream("libx264", rate=in_video.average_rate or 30)
    out_video.width = in_video.width
    out_video.height = in_video.height
    out_video.pix_fmt = "yuv420p"  # Mandatory for HTML5 web browser playback
    out_video.options = {"crf": "23", "preset": "fast"}

    frame_count = 0
    for frame in input_container.decode(in_video):
        for packet in out_video.encode(frame):
            output_container.mux(packet)
        frame_count += 1

    # Flush encoder
    for packet in out_video.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()

    print(f"[SUCCESS] Re-encoded {frame_count} frames to H.264 at: {output_path}")
    return True


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/real_demo.mp4"
    dst = sys.argv[2] if len(sys.argv) > 2 else "data/real_demo_h264.mp4"

    success = reencode_to_h264(src, dst)
    if success and src == "data/real_demo.mp4" and dst == "data/real_demo_h264.mp4":
        # Replace original with the web-compatible H.264 version
        import time
        time.sleep(0.5)
        os.replace(dst, src)
        print(f"[SUCCESS] Replaced '{src}' with browser-compatible H.264 stream.")

