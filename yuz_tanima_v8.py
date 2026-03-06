from ultralytics import YOLO
import cv2
import time

IP_CAMERA_URL = "http://192.168.1.128:8080/video" 
MODEL_NAME = 'yolov8n.pt' 

CONFIDENCE_THRESHOLD = 0.5

try:

    model = YOLO(MODEL_NAME)
    print(f"YOLOv8 Nano modeli başarıyla yüklendi: {MODEL_NAME}")
except Exception as e:
    print(f"HATA: YOLOv8 modelini yüklerken bir sorun oluştu. Ultralytics kurulumunu kontrol edin: {e}")
    exit()

cap = cv2.VideoCapture(IP_CAMERA_URL)

if not cap.isOpened():
    print(f"HATA: IP Akışı açılamadı. URL'yi kontrol edin: {IP_CAMERA_URL}")
    exit()

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("UYARI: Kare (frame) okunamadı. Akış kesilmiş olabilir.")
        time.sleep(0.1) 
        continue
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,

        imgsz=416 
    )

    annotated_frame = results[0].plot()

    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        start_time = time.time()

    cv2.imshow('YOLOv8 Nano Nesne Tanima', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()

