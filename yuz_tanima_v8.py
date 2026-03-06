from ultralytics import YOLO
import cv2
import time

# --- 1. AYARLAR ---
# IP Webcam'den aldığınız URL'yi BURAYA yazın.
# Örn: "http://192.168.1.120:8080/video"
IP_CAMERA_URL = "http://192.168.1.128:8080/video" 

# YOLOv8 nano modelini yükle (ilk çalıştırmada otomatik indirilir)
# n = nano, s = small, m = medium, l = large
MODEL_NAME = 'yolov8n.pt' 

# Güven eşiği (0.25 varsayılandır, daha az güvenli tespiti göstermek için düşürebilirsiniz)
CONFIDENCE_THRESHOLD = 0.5

# --- 2. MODELİ YÜKLEME ---
try:
    # Model otomatik olarak indirilir ve GPU/CPU'da çalışmaya ayarlanır.
    model = YOLO(MODEL_NAME)
    print(f"YOLOv8 Nano modeli başarıyla yüklendi: {MODEL_NAME}")
except Exception as e:
    print(f"HATA: YOLOv8 modelini yüklerken bir sorun oluştu. Ultralytics kurulumunu kontrol edin: {e}")
    exit()

# --- 3. CANLI GÖRÜNTÜ İŞLEME ---
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
        
    # --- YOLOv8 ile Nesne Tespiti ---
    # predict() fonksiyonu ile tespiti gerçekleştir
    # conf=CONFIDENCE_THRESHOLD: Güven eşiği
    # verbose=False: Konsola ayrıntılı çıktı vermeyi kapat
    results = model.predict(
        source=frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        # Çözünürlüğü zorlamak için (önerilir, hızı artırır)
        imgsz=416 
    )

    # Tespit sonuçlarını görselleştir
    # results[0].plot() fonksiyonu, tespiti otomatik olarak çizer
    # ve etiketi (insan, hayvan, yemek vb.) kare üzerine yazar.
    annotated_frame = results[0].plot()

    # --- FPS Hesaplama (İsteğe Bağlı) ---
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame_count = 0
        start_time = time.time()
        
    # Sonucu ekranda göster
    cv2.imshow('YOLOv8 Nano Nesne Tanima', annotated_frame)
    
    # 'q' tuşuna basılırsa döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()