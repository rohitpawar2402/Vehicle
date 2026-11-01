from collections import deque
import threading
import time
from typing import Optional, Tuple, Union
import cv2
import numpy as np


class StreamBuffer:
    """Threaded RTSP / Video capture buffer that continuously drains frames

    into a fixed-length FIFO queue to eliminate network lag, packet dropping,
    and decode latency in live CCTV streams.
    """

    def __init__(
        self,
        source: Union[str, int],
        buffer_size: int = 2,
        reconnect_delay: float = 2.0,
    ):
        self.source = int(source) if str(source).isdigit() else str(source)
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.queue: deque = deque(maxlen=buffer_size)

        self.stopped = False
        self.is_opened = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self.fps = 30
        self.width = 1280
        self.height = 720

        # Initial source capture
        self.cap: Optional[cv2.VideoCapture] = cv2.VideoCapture(self.source)
        if self.cap.isOpened():
            self.is_opened = True
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    def start(self) -> "StreamBuffer":
        self.stopped = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self) -> None:
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                if isinstance(self.source, str) and (
                    self.source.startswith("rtsp://") or self.source.startswith("http://")
                ):
                    print(f"[WARN] Connection lost to {self.source}. Reconnecting in {self.reconnect_delay}s...")
                    time.sleep(self.reconnect_delay)
                    self.cap = cv2.VideoCapture(self.source)
                    continue
                else:
                    break

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Video file reached EOF or stream stalled
                if not (isinstance(self.source, str) and self.source.startswith("rtsp://")):
                    break
                time.sleep(0.01)
                continue

            with self.lock:
                self.queue.append(frame)

        self.is_opened = False

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieves the freshest available frame from the buffer, dropping older frames."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if self.queue:
                    return True, self.queue.pop()
            if not self.is_opened and not self.queue:
                return False, None
            time.sleep(0.005)

        return False, None

    def stop(self) -> None:
        self.stopped = True
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()

