import base64
import cv2
import requests
import time

from yolov7 import YOLOv7

theSecret = "Backend secret"

# Initialise the webcam
cap = cv2.VideoCapture(0)

num = '1'

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
frame_rate_calc = 1
freq = cv2.getTickFrequency()
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
scores = []

while cap.isOpened():
    try:
        t1 = cv2.getTickCount()

        if len(scores) > 0 and ( num == '1' or num == '2' ):
            num = '2'

            if boxes[0][0] > 5 and boxes[0][1] > 5 and boxes[0][2] < frame.shape[1] - 5 and boxes[0][3] < frame.shape[0] - 5:
                ret, buffer = cv2.imencode('.jpg', frame)
                string = base64.b64encode(buffer).decode('utf-8')
                my_img = {
                    'image': string,
                    'number': 5,
                    'secret': theSecret,
                }

                # Send frame to Backend
                res = requests.post('https://ll753-flaskmlbackendlpr.hf.space/sendlicenseY', json=my_img)
                res.raise_for_status()
                print(res.content)
                num = '3'
                if not 'Image' in res.content:
                    num = '1'

        elif num == '3' and len(scores) == 0:
            num = '1'

        # Read frame from video
        ret, frame = cap.read()

        if not ret:
            break

        # Update object localizer
        boxes, scores, class_ids = yolov7_detector(frame)
        # print(boxes, scores, class_ids)

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
    except:
        continue

cap.release()
cv2.destroyAllWindows()