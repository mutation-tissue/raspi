import cv2
import numpy as np
import torch
from picamera.array import PiRGBArray
from picamera import PiCamera

# YOLOv5モデルをロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# カメラの初期設定
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

print("Press Ctrl+C to stop")

try:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # フレームを取得
        image = frame.array

        # YOLOv5で物体検出
        results = model(image)

        # 検出結果のバウンディングボックスを取得
        for *box, conf, cls in results.xyxy[0].tolist():
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"

            # バウンディングボックスを描画
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 映像を表示
        cv2.imshow("Camera with Object Detection", image)

        # フレームバッファをクリア
        rawCapture.truncate(0)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by User")

finally:
    # リソースを解放
    cv2.destroyAllWindows()
    camera.close()
