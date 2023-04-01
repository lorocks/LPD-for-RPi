import cv2

from yolov7 import YOLOv7

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv7 object detector
model_path = "models/yolov7-tiny.onnx"
yolov7_detector = YOLOv7(model_path, conf_thres=0.5, iou_thres=0.5)
frame_rate_calc = 1
freq = cv2.getTickFrequency()
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    t1 = cv2.getTickCount()
    ret, frame = cap.read()

    # Read frame from the video
    ret, frame = cap.read()

    if not ret:
        break

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