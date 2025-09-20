import cv2
from core.detection import Detection

class VideoStream:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = Detection()

    def generate(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Run detection
            detections = self.detector.detect(frame)
            frame = self.detector.draw_detections(frame, detections)

            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
