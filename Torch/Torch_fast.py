import torch
import cv2
import time
from threading import Thread

# model = torch.hub.load('yolov7', 'custom', 'YOLOv7pt/best.pt', source="local")
# model = torch.hub.load('yolov7', 'custom', 'YOLOv7pt/yolov7.onnx', source="local")

# For YOLO Tiny Model
model = torch.hub.load('yolov7', 'custom', 'YOLOv7 Tiny/best.pt', source="local")
# model = model.zero_grad()
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

frame_rate_calc = 1
freq = cv2.getTickFrequency()


videostream = VideoStream().start()
time.sleep(1)
while True:
    t1 = cv2.getTickCount()
    frame = videostream.read()

    with torch.no_grad(): #this i added to check
        results = model(frame)
    image = frame
    print(results.pandas().xyxy[0])

    if len(results.pandas().xyxy[0]) != 0:
        rows = results.pandas().xyxy[0].shape[0]
        for i in range(rows):
            xmin = int(results.pandas().xyxy[0].iloc[i].xmin)
            xmax = int(results.pandas().xyxy[0].iloc[i].xmax)
            ymin = int(results.pandas().xyxy[0].iloc[i].ymin)
            ymax = int(results.pandas().xyxy[0].iloc[i].ymax)

            cv2.rectangle(image,(xmin, ymin),(xmax, ymax),(0,255,0),3)
            cv2.putText(image,str(results.pandas().xyxy[0].confidence.get(i)),(xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255),2,cv2.LINE_AA)

    cv2.putText(image, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("LP Image", image)
    t2 = cv2.getTickCount()
    time1 = (t2 - t1) / freq
    frame_rate_calc = 1 / time1


    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()