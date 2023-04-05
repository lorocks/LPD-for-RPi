import cv2

from yolov7 import YOLOv7
import time
from threading import Thread

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution=(640, 480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3, resolution[0])
        ret = self.stream.set(4, resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Initialize the webcam
videostream = VideoStream().start()
time.sleep(1)

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
frame_rate_calc = 1
freq = cv2.getTickFrequency()
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while True:
    t1 = cv2.getTickCount()

    # Read frame from the video
    frame = videostream.read()

    # Update object localizer
    boxes, scores, class_ids = yolov7_detector(frame)
    print(boxes, scores, class_ids)

    combined_img = yolov7_detector.draw_detections(frame)
    cv2.putText(combined_img, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("Detected Objects", combined_img)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1

    # Press key q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()