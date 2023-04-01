import torch
import cv2

# model = torch.hub.load('yolov7', 'custom', 'YOLOv7pt/best.pt', source="local")
# model = torch.hub.load('yolov7', 'custom', 'YOLOv7pt/yolov7.onnx', source="local")

# For YOLO Tiny Model
model = torch.hub.load('yolov7', 'custom', 'YOLOv7 Tiny/best.pt', source="local")
# model = model.zero_grad()

frame_rate_calc = 1
freq = cv2.getTickFrequency()


cap = cv2.VideoCapture(0)
while cap.isOpened():
    t1 = cv2.getTickCount()
    ret, frame = cap.read()

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
        cap.release()
        cv2.destroyAllWindows()