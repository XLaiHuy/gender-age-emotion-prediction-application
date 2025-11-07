import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog,
    QHBoxLayout, QFrame, QGraphicsDropShadowEffect, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
from PyQt5.QtCore import QTimer, Qt


# =========================
#       MODEL FILES
# =========================
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4, 87.76, 114.89)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']


def detect_faces(face_net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    det = face_net.forward()
    boxes = []
    for i in range(det.shape[2]):
        conf = det[0, 0, i, 2]
        if conf > 0.7:
            x1 = int(det[0, 0, i, 3] * w)
            y1 = int(det[0, 0, i, 4] * h)
            x2 = int(det[0, 0, i, 5] * w)
            y2 = int(det[0, 0, i, 6] * h)
            boxes.append([x1, y1, x2, y2])
    return boxes


# =========================
#          UI
# =========================
class AgeGenderApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Age & Gender Detector — Windows 11 UI")
        self.resize(1280, 780)

        # State
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.current_frame = None
        self.dark_mode = False

        # Title
        self.title = QLabel("Age & Gender Detector")
        self.title.setFont(QFont("Segoe UI Variable Display", 32, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)

        # Card
        self.card = QFrame()
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 15)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.card.setGraphicsEffect(shadow)

        self.image_label = QLabel()
        self.image_label.setFixedSize(900, 520)
        self.image_label.setAlignment(Qt.AlignCenter)

        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(30, 30, 30, 30)
        card_layout.addWidget(self.image_label)
        self.card.setLayout(card_layout)

        # Buttons
        self.btn_load = self.new_btn("📁 Load")
        self.btn_start = self.new_btn("🎥 Start")
        self.btn_stop = self.new_btn("⛔ Stop")
        self.btn_save = self.new_btn("💾 Save")
        self.btn_theme = self.new_btn("🌓 Theme")

        self.btn_load.clicked.connect(self.load_image)
        self.btn_start.clicked.connect(self.start_webcam)
        self.btn_stop.clicked.connect(self.stop_webcam)
        self.btn_save.clicked.connect(self.save_image)
        self.btn_theme.clicked.connect(self.toggle_theme)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        for b in [self.btn_load, self.btn_start, self.btn_stop, self.btn_save, self.btn_theme]:
            btn_row.addWidget(b)
            btn_row.addSpacing(14)
        btn_row.addStretch()

        # Layout ROOT
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 20, 40, 20)
        root.addWidget(self.title)
        root.addWidget(divider)
        root.addSpacing(8)

        # ✅ Center webcam card
        center_box = QHBoxLayout()
        center_box.addStretch()
        center_box.addWidget(self.card)
        center_box.addStretch()
        root.addLayout(center_box)

        root.addSpacing(14)
        root.addLayout(btn_row)

        # Apply theme
        self.setWindowOpacity(1.0)
        self.apply_theme(light=True)

    # ========================================
    #           THEME ENGINE
    # ========================================
    def apply_theme(self, light=True):

        if light:
            # Background Fluent Light
            self.setStyleSheet("""
                QWidget {
                    background: qlineargradient(x1:0 y1:0, x2:1 y2:1,
                        stop:0 #F6F8FB, stop:1 #E8ECF3);
                    color: #1B1D21;
                    font-family: 'Segoe UI Variable Display';
                }
            """)

            self.title.setStyleSheet("color:#1A1A1C; padding:20px;")

            self.card.setStyleSheet("""
                QFrame {
                    background: rgba(255,255,255,0.65);
                    border-radius: 26px;
                    border: 1px solid rgba(180,180,180,0.35);
                }
            """)

            self.image_label.setStyleSheet("""
                QLabel {
                    background: rgba(255,255,255,0.75);
                    border-radius: 18px;
                    border: 1px solid rgba(200,200,200,0.45);
                }
            """)

            btn_style = self.btn_style_light()

        # ================= DARK THEME =================
        else:
            self.setStyleSheet("""
                QWidget {
                    background: qlineargradient(x1:0 y1:0, x2:1 y2:1,
                        stop:0 #0D0E11, stop:1 #1A1C20);
                    color:#EEE;
                    font-family: 'Segoe UI Variable Display';
                }
            """)

            self.title.setStyleSheet("color:#EEE; padding:20px;")

            self.card.setStyleSheet("""
                QFrame {
                    background: rgba(32,34,40,0.55);
                    border-radius: 26px;
                    border: 1px solid rgba(255,255,255,0.08);
                }
            """)

            self.image_label.setStyleSheet("""
                QLabel {
                    background: rgba(18,20,24,0.75);
                    border-radius:18px;
                    border:1px solid rgba(255,255,255,0.06);
                }
            """)

            btn_style = self.btn_style_dark()

        for b in [self.btn_load, self.btn_start, self.btn_stop, self.btn_save, self.btn_theme]:
            b.setStyleSheet(btn_style)

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.apply_theme(light=not self.dark_mode)

    # ========================================
    #                 BUTTONS
    # ========================================
    def new_btn(self, text):
        b = QPushButton(text)
        b.setMinimumHeight(40)
        b.setCursor(Qt.PointingHandCursor)
        return b

    def btn_style_light(self):
        return """
            QPushButton {
                background: #FFFFFF;
                border: 1px solid #D0D3DA;
                border-radius: 12px;
                padding: 10px 18px;
                font-size: 15px;
                color: #1B1D21;
            }
            QPushButton:hover { background:#F2F3F7; }
            QPushButton:pressed { background:#E4E6EB; }
        """

    def btn_style_dark(self):
        return """
            QPushButton {
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius:12px;
                padding:10px 18px;
                font-size:15px;
                color:#EEE;
            }
            QPushButton:hover { background:rgba(255,255,255,0.18); }
            QPushButton:pressed { background:rgba(255,255,255,0.25); }
        """

    # ========================================
    #                  LOGIC
    # ========================================
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        img = cv2.imread(path)
        self.current_frame = img.copy()
        self.render_and_show(img)

    def start_webcam(self):
        if self.cap:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.timer.start(30)

    def stop_webcam(self):
        if self.cap:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.image_label.clear()

    def save_image(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Save", "⚠ Nothing to save.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Image", "image.png",
                                              "PNG (*.png);; JPG (*.jpg)")
        if not path:
            return

        cv2.imwrite(path, self.current_frame)
        QMessageBox.information(self, "Saved", f"✅ Saved to:\n{path}")

    def update_frame(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        self.render_and_show(frame)

    # ========================================
    #                 RENDER
    # ========================================
    def render_and_show(self, frame_bgr):
        annotated = self.render_only(frame_bgr.copy())
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self.image_label.width(),
                         self.image_label.height(),
                         Qt.KeepAspectRatio,
                         Qt.SmoothTransformation)
        self.image_label.setPixmap(pix)

    def render_only(self, frame):
        boxes = detect_faces(faceNet, frame)
        for (x1, y1, x2, y2) in boxes:

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            gender = genderList[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob)
            age = ageList[ageNet.forward()[0].argmax()]

            label = f"{gender}, {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        return frame


# =========================
#          RUN
# =========================
app = QApplication(sys.argv)
win = AgeGenderApp()
win.show()
sys.exit(app.exec_())
