import cv2
import numpy as np
import time

# --- 1. DOSYA YOLLARINI TANIMLAMA ---
# Bu dosyaların Python kodunuzla aynı klasörde olduğundan emin olun!
MODEL_CFG = 'yolov3.cfg'
MODEL_WEIGHTS = 'yolov3.weights'
CLASSES_FILE = 'coco.names'

# Güven eşiği (bu değerden düşük doğruluktaki nesneler göz ardı edilir)
CONFIDENCE_THRESHOLD = 0.5
# Non-Maximum Suppression (Çakışan kutuları azaltır) eşiği
NMS_THRESHOLD = 0.4
# YOLO modelinin beklediği giriş boyutu
INPUT_WIDTH = 416
INPUT_HEIGHT = 416

# --- 2. SINIF İSİMLERİNİ YÜKLEME ---
try:
    with open(CLASSES_FILE, 'r') as f:
        # Her satırdan sınıf adını alıp listede sakla
        classes = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    print(f"HATA: '{CLASSES_FILE}' dosyası bulunamadı. Lütfen gerekli dosyaları indirdiğinizden emin olun.")
    exit()

# Her sınıf için rastgele bir renk oluşturma (Görselleştirme için)
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# --- 3. AĞIRLIK VE KONFİGÜRASYON DOSYALARINI YÜKLEME ---
try:
    net = cv2.dnn.readNet(MODEL_WEIGHTS, MODEL_CFG)
    # OpenCV'nin DNN modülünü kullan
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) # CPU'da çalışmayı tercih et (GPU varsa GPU da kullanılabilir)
except cv2.error as e:
    print(f"HATA: YOLO dosyalarını yüklerken bir sorun oluştu. Dosya adlarını kontrol edin: {e}")
    exit()

# Çıktı katmanlarının adlarını al (YOLO'nun sonuçlarını veren katmanlar)
layer_names = net.getLayerNames()
# Yeni OpenCV sürümlerinde bu biraz farklı çalışır:
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


# --- 4. YARDIMCI İŞLEV: TESPİT SONUÇLARINI İŞLEME ---
def process_detections(frame, outputs):
    height, width, _ = frame.shape
    boxes = []
    confidences = []
    class_ids = []

    # Her çıktı katmanındaki her bir tespiti döngüye al
    for output in outputs:
        for detection in output:
            # Sınıf olasılıklarını al (ilk 5 eleman bbox bilgileri, sonrası sınıflar)
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Belirlenen eşiğin üzerindeki tespitleri filtrele
            if confidence > CONFIDENCE_THRESHOLD:
                # Nesnenin merkez koordinatlarını ve boyutlarını al
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Dikdörtgenin sol üst köşesini hesapla
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Çakışan kutuları (bbox) kaldırmak için Non-Maximum Suppression (NMS) uygula
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    # Kalan (en iyi) tespitleri çiz
    for i in indexes:
        # i'nin bir skalar veya tek elemanlı bir dizi olup olmadığını kontrol et
        if isinstance(i, np.ndarray):
            i = i[0]
        
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        
        # Etiket ve güven skorunu birleştir
        text = f"{label}: {confidence:.2f}"
        
        # Sınıfın rengini al
        color = [int(c) for c in colors[class_ids[i]]]
        
        # Dikdörtgeni ve etiketi çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# --- 5. CANLI GÖRÜNTÜ İŞLEME ---
ip_kamera_url = "http://192.168.1.120:8080/video" 

# Kamerayı IP adresi üzerinden başlat
cap = cv2.VideoCapture(ip_kamera_url)

if not cap.isOpened():
    print(f"HATA: IP Akışı açılamadı. URL'yi kontrol edin: {ip_kamera_url}")
    exit()

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Görüntüyü YOLO'nun beklediği formata dönüştür (Blob oluşturma)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    
    # Blob'u ağa giriş olarak ver ve ileri besleme (forward pass) yap
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Tespit sonuçlarını işle ve çiz
    process_detections(frame, outputs)
    
    # Sonucu ekranda göster
    cv2.imshow('Kapsamli Nesne Tanima (YOLOv3)', frame)
    
    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()