import torch
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import cv2
import os
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5 import QtGui
from PyQt5.QtCore import *
from fruit import Ui_MainWindow

# Load mô hình đã được huấn luyện
model = resnet18(num_classes=5)
model.load_state_dict(torch.load('fruit_classifier_resnet2.pth'))
model.eval()
# Danh sách nhãn biển báo
predicted_label = ["apple", "banana", "mango", "orange", "strawberry"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.uic.btn_Start.clicked.connect(self.start_capture_video)
        self.uic.btn_stop.clicked.connect(self.stop_capture_video)
        self.image_folder = "images" 
        os.makedirs(self.image_folder, exist_ok=True)
        self.cap = None

    def start_capture_video(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        while True:
            # Đọc khung hình từ camera
            ret, frame = self.cap.read()
            # Chuyển đổi khung hình sang định dạng PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Tiền xử lý ảnh
            image = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(image)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                max_prob, predicted_index = torch.max(probabilities, dim=0)

            # Kiểm tra nếu giá trị xác suất của dự đoán cao nhất dưới ngưỡng 0.6 thì coi là không xác định
            if max_prob < 0.5:
                predicted_label_text = "unknown"
            else:
                predicted_label_text = predicted_label[predicted_index]

            print(predicted_label_text)
            
            if predicted_label_text == "apple":
                self.uic.txt_name.setText('Đây là trái TÁO')
                
            elif predicted_label_text == "banana":   
                self.uic.txt_name.setText('Đây là trái CHUỐI')
                
            elif predicted_label_text == "mango":   
                self.uic.txt_name.setText('Đây là trái XOÀI')
                       
            elif predicted_label_text == "orange":
                self.uic.txt_name.setText('Đây là trái CAM')
                                  
            elif predicted_label_text == "strawberry":  
                self.uic.txt_name.setText('Đây là trái DÂU')
                
            else:
                self.uic.txt_name.setText('Không xác định!')
                                    
            qt_img = self.convert_cv_qt(frame)
            self.uic.lb_camera.setPixmap(qt_img)
            # Nhấn phím 'q' để thoát khỏi vòng lặp
            if cv2.waitKey(1) == ord('q'):
                break

    def convert_cv_qt(self, cv_img):
            if cv_img is not None:
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
                return QPixmap.fromImage(p)
            else:
                return None
         
    def closeEvent(self, event):
        self.stop_capture_video()

    def stop_capture_video(self):
        message = QMessageBox.warning(self, "Warning", "Do you really want to exit?", QMessageBox.Yes | QMessageBox.No)
        if message == QMessageBox.Yes:
            if self.cap is not None:
                self.cap.release()
            sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

cv2.destroyAllWindows()
