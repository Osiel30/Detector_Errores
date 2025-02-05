import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QFileDialog)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

#  FUNCION PARA MEJORAR DETECCIÓN (Por pigmentación)
def preprocess_by_pigmentation(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar pigmentación mediante umbral adaptativo
    pigment_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 10)

    # Operación morfológica para cerrar pequeños huecos en los números
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(pigment_mask, cv2.MORPH_CLOSE, kernel)

    return processed

#  CLASE PRINCIPAL
class MathValidator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Validador de Multiplicaciones")
        self.setFixedSize(1000, 700)
        self.image = None
        self.image_path = None
        self.templates = self.load_templates('mis_numeros')
        self.init_ui()

    #  Carga de plantillas de números
    def load_templates(self, template_folder):
        templates = {}
        files = {str(i): f'num_{i}.jpg' for i in range(10)}
        files['x'] = 'num_x.png'

        for label, filename in files.items():
            path = os.path.join(template_folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"⚠️ Error al cargar la plantilla: {path}")
            else:
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                templates[label] = img_bin
        return templates

    #  Interfaz gráfica
    def init_ui(self):
        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()

        self.btn_load = QPushButton("Cargar Imagen")
        self.btn_process = QPushButton("Procesar")

        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_process)

        self.lbl_original = QLabel()
        self.lbl_processed = QLabel()
        self.lbl_result = QLabel("Resultado aparecerá aquí")
        self.lbl_detected = QLabel("Dígitos detectados: ")

        for lbl in [self.lbl_original, self.lbl_processed]:
            lbl.setFixedSize(400, 300)
            lbl.setStyleSheet("border: 2px solid #333; background: #f8f8f8;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.lbl_original)
        main_layout.addWidget(self.lbl_processed)
        main_layout.addWidget(self.lbl_result)
        main_layout.addWidget(self.lbl_detected)

        self.setLayout(main_layout)

        self.btn_load.clicked.connect(self.load_image)
        self.btn_process.clicked.connect(self.process_image)

    #  Cargar imagen
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir Imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if path:
            self.image_path = path
            self.image = cv2.imread(path)
            if self.image is None:
                self.lbl_result.setText("⚠️ Error al cargar la imagen.")
                return
            self.display_image(self.image, self.lbl_original)
            self.lbl_result.setText("✅ Imagen cargada.")
            self.lbl_detected.setText("Dígitos detectados: ")

    #  PREPROCESAMIENTO DE LA IMAGEN (Usa la nueva función basada en pigmentación)
    def preprocess_image(self, image):
        return preprocess_by_pigmentation(image)

    #  DETECTAR REGIONES DE LOS NÚMEROS
    def detect_components(self, processed, min_area=100):
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        components = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h > min_area:
                roi = processed[y:y+h, x:x+w]
                components.append((x, y, w, h, roi))
        components = sorted(components, key=lambda c: (c[1], c[0]))  # Ordenar de izq a der
        return components

    #  CLASIFICAR DÍGITO DETECTADO
    def classify_character(self, roi):
        best_label = '?'
        best_score = -1
        _, roi_bin = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

        for label, temp in self.templates.items():
            try:
                roi_resized = cv2.resize(roi_bin, (temp.shape[1], temp.shape[0]))
            except Exception:
                continue
            res = cv2.matchTemplate(roi_resized, temp, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]
            if score > best_score:
                best_score = score
                best_label = label
        return best_label, best_score

    #  PROCESAR IMAGEN PARA DETECTAR NÚMEROS
    def process_image(self):
        if self.image is None:
            self.lbl_result.setText("⚠️ ¡Primero carga una imagen!")
            return
        
        processed = self.preprocess_image(self.image)
        components = self.detect_components(processed)

        detected_text = ""
        output_image = self.image.copy()

        for (x, y, w, h, roi) in components:
            label, score = self.classify_character(roi)
            detected_text += label + " "
            cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output_image, label, (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        self.display_image(output_image, self.lbl_processed)
        self.lbl_detected.setText(f"Dígitos detectados: {detected_text.strip() if detected_text.strip() else 'Ninguno detectado'}")
        self.lbl_result.setText("✅ Procesamiento completado.")

    #  MOSTRAR IMAGEN EN UI
    def display_image(self, image, label):
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, channel = image_rgb.shape
            bytes_per_line = 3 * width
            q_img = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img).scaled(label.width(), label.height(),
                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
        label.setPixmap(pixmap)

#  EJECUCIÓN DE LA APLICACIÓN
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MathValidator()
    window.show()
    sys.exit(app.exec())