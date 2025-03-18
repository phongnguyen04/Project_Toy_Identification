from ultralytics import YOLO
import cv2
from urllib.request import urlopen
import numpy as np
import time
import serial
import threading

# Khởi tạo mô hình YOLO
model = YOLO("best.pt")
url = 'http://192.168.100.39/cam-hi.jpg'
FRUIT_CLASSES = {0: "car", 1: "toys"}

# Kết nối Arduino
try:
    arduino = serial.Serial(port="COM3", baudrate=115200, timeout=1)
    time.sleep(2)
    print("Connected to Arduino")
except Exception as e:
    print(f"Arduino connection error: {e}")
    arduino = None

# Hàm lấy ảnh từ ESP32-CAM
def get_frame_from_esp():
    try:
        img_resp = urlopen(url, timeout=2)
        imgnp = np.asarray(bytearray(img_resp.read()), dtype=np.uint8)
        return cv2.imdecode(imgnp, -1)
    except Exception as e:
        print(f"Camera error: {e}")
        return None

# Hàm xử lý yêu cầu từ Arduino
def handle_arduino_request():
    while True:
        if arduino and arduino.in_waiting > 0:
            try:
                message = arduino.readline().decode().strip()
                if message == "CHECK":
                    print("\n🔎 Received CHECK request")
                    
                    # Chụp ảnh và nhận diện
                    frame = get_frame_from_esp()
                    
                    if frame is not None:
                        results = model(frame, conf=0.3)  # Giảm conf threshold xuống 0.3
                        result_class = "toys"  # Mặc định nếu không phát hiện vật thể nào

                        print("Raw YOLO results:")
                        for result in results:
                            print(f"Detected boxes: {result.boxes}")  # Debug YOLO output

                        # Vẽ bounding box và phân tích kết quả
                        for result in results:
                            if len(result.boxes) > 0:  # Kiểm tra có vật thể không
                                for box in result.boxes:
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ hộp
                                    class_id = int(box.cls[0])
                                    label = FRUIT_CLASSES.get(class_id, "Unknown")

                                    # Vẽ khung nhận diện
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, (0, 255, 0), 2)

                                    print(f"Detected class_id: {class_id} -> {label}")  # Debug nhận diện
                                    result_class = label
                                    break

                        # Gửi kết quả về Arduino
                        response = f"{result_class}\n"
                        arduino.write(response.encode())
                        print(f"Sent to Arduino: {result_class}")

                        # Lưu ảnh để kiểm tra (nếu cần)
                        cv2.imwrite("captured_image.jpg", frame)

                    else:
                        print("Failed to capture image")
                        arduino.write("toys\n".encode())
            except Exception as e:
                print(f"Serial error: {e}")

# Khởi chạy luồng xử lý Serial
serial_thread = threading.Thread(target=handle_arduino_request)
serial_thread.daemon = True
serial_thread.start()

# Chạy vòng lặp chính (Hiển thị Camera)
print("System Ready")
while True:
    frame = get_frame_from_esp()
    
    if frame is not None:
        # Chạy YOLO trên hình ảnh để hiển thị bounding box
        results = model(frame, conf=0.3)

        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ hộp
                    class_id = int(box.cls[0])
                    label = FRUIT_CLASSES.get(class_id, "Unknown")

                    # Vẽ khung nhận diện
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 255, 0), 2)

                    print(f"Detected class_id: {class_id} -> {label}")  # Debug nhận diện

        cv2.imshow("ESP32-CAM Stream", frame)  # Hiển thị hình ảnh có nhận diện
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn 'q' để thoát
        break

cv2.destroyAllWindows()  # Đóng cửa sổ khi thoát
