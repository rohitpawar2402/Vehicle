import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ingestion.stream_buffer import StreamBuffer


def test_stream_buffer_video():
    video_path = "data/sample_traffic.mp4"
    if not os.path.exists(video_path):
        print("[SKIP] data/sample_traffic.mp4 not found.")
        return

    buf = StreamBuffer(source=video_path, buffer_size=2)
    assert buf.is_opened is True
    assert buf.width == 1280 and buf.height == 720

    buf.start()
    frames_read = 0
    for _ in range(10):
        ret, frame = buf.read(timeout=1.0)
        if ret and frame is not None:
            frames_read += 1
            assert frame.shape == (720, 1280, 3)

    buf.stop()
    assert frames_read > 0, "Failed to read any frames from StreamBuffer"
    print(f"[OK] StreamBuffer successfully ingested {frames_read} frames from video source!")


if __name__ == "__main__":
    test_stream_buffer_video()
    print("\nAll StreamBuffer tests passed successfully!")

