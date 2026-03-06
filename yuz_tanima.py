import cv2
import numpy as np
import time

MODEL_CFG = 'yolov3.cfg'
MODEL_WEIGHTS = 'yolov3.weights'
CLASSES_FILE = 'coco.names'

CONFIDENCE_THRESHOLD = 0.5

NMS_THRESHOLD = 0.4

INPUT_WIDTH = 416
INPUT_HEIGHT = 416


try:
    with open(CLASSES_FILE, 'r') as f:
        
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"HATA: '{CLASSES_FILE}' dosyası bulunamadı. Lütfen gerekli dosyaları indirdiğinizden emin olun.")
    exit()


np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')


try:
    net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CFG)
    
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
except cv2.error as e:
    print(f"HATA: YOLO dosyalarını yüklerken bir sorun oluştu. Dosya adlarını kontrol edin: {e}")
    exit()


layer_names = net.getLayerNames()

output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def process_detections(frame, outputs):
    height, width, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []

   
    for output in outputs:
        for detection in output:
           
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            
            if confidence > CONFIDENCE_THRESHOLD:
                
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

   
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    
    for i in indexes:
       
        if isinstance(i, np.ndarray):
            i = i[0]
        
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
      
        text = f"{label}: {confidence:.2f}"
        
        color = [int(c) for c in colors[class_ids[i]]]

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

ip_kamera_url = "http://192.168.1.120:8080/video" 

cap = cv2.VideoCapture(ip_kamera_url)

if not cap.isOpened():
    print(f"HATA: IP Akışı açılamadı. URL'yi kontrol edin: {ip_kamera_url}")
    exit()

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(output_layers)

    process_detections(frame, outputs)
    
    cv2.imshow('Kapsamli Nesne Tanima (YOLOv3)', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
