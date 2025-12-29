import sys
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QCheckBox,
)
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtCore import Qt, pyqtSlot

from fly_dl.GUI.mainWindow import Ui_MainWindow
from fly_dl.core.detector import YoloThread


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 1. 初始化 UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 设置模型权重的绝对路径
        self.weights_dir = r"D:\fly\fly_dl\model\weights"

        # 2. 初始化变量
        self.current_frame = None
        self.detector = YoloThread()

        # 3. 动态注入 OpenGL 绘图逻辑
        self.ui.openGLWidget.paintEvent = self.opengl_paint_event

        # 4. 添加跟踪功能复选框
        self.add_tracking_checkbox()

        # 5. 连接信号与槽
        self.bind_signals()

        # 6. 初始化控件与模型列表
        self.init_model_list()  # 扫描文件夹
        self.init_controls()

        #  默认加载列表中的第一个模型
        self.on_model_change()

    def add_tracking_checkbox(self):
        """动态添加跟踪功能复选框"""
        self.tracking_checkbox = QCheckBox("启用跟踪", self.ui.groupBox_settings)
        self.tracking_checkbox.setGeometry(150, 20, 100, 20)
        self.tracking_checkbox.setObjectName("checkBox_tracking")
        # 连接复选框状态变化信号
        self.tracking_checkbox.stateChanged.connect(self.on_tracking_state_changed)

    def init_model_list(self):
        """扫描指定目录下的 .pt 文件并填充到下拉框"""
        self.ui.comboBox_select_model.clear()  # 先清空 UI 文件里写死的假数据

        if not os.path.exists(self.weights_dir):
            print(f"错误：路径不存在 -> {self.weights_dir}")
            self.ui.comboBox_select_model.addItem("路径错误")
            return

        # 扫描目录下的 .pt 文件
        model_files = [f for f in os.listdir(self.weights_dir) if f.endswith(".pt")]

        if model_files:
            self.ui.comboBox_select_model.addItems(model_files)
            print(f"已加载模型列表: {model_files}")
        else:
            self.ui.comboBox_select_model.addItem("无模型文件")
            print("指定目录下没有找到 .pt 文件")

    def init_controls(self):
        # 初始化滑块和 Label
        initial_conf = 50
        self.ui.horizontalSlider_adjust_confidence.setMaximum(100)
        self.ui.horizontalSlider_adjust_confidence.setValue(initial_conf)
        self.ui.label_confidence_num.setText("0.50")
        self.detector.update_config(0.5)

        # 初始化字体 Dial
        self.ui.dial_detection_font_setting.setRange(1, 10)
        self.ui.dial_detection_font_setting.setValue(2)
        self.ui.label_font_size.setText("2")

    def bind_signals(self):
        # --- 图像/视频处理信号 ---
        self.detector.frame_processed.connect(self.update_display)

        # 连接模型加载完成信号
        self.detector.model_loaded.connect(self.on_model_loaded)

        # --- 按钮控制 ---
        self.ui.pushButton_open_camera.clicked.connect(
            lambda: self.start_detection(source=0)
        )
        self.ui.pushButton_close_camera.clicked.connect(self.stop_detection)

        self.ui.pushButton_select_video.clicked.connect(self.select_video_file)
        self.ui.pushButton_detect_video.clicked.connect(self.start_video_detection)
        self.ui.pushButton_stop_display.clicked.connect(self.stop_detection)

        self.ui.pushButton_select_picture.clicked.connect(self.select_image_file)
        self.ui.pushButton_detect_picture.clicked.connect(self.detect_image)

        # --- 设置控制 ---
        self.ui.horizontalSlider_adjust_confidence.valueChanged.connect(
            self.on_conf_change
        )
        self.ui.dial_detection_font_setting.valueChanged.connect(
            lambda val: self.ui.label_font_size.setText(str(val))
        )
        # 下拉框变化时触发模型重新加载
        self.ui.comboBox_select_model.currentIndexChanged.connect(self.on_model_change)

    # --- 核心功能实现 ---

    def opengl_paint_event(self, event):
        painter = QPainter(self.ui.openGLWidget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.ui.openGLWidget.rect(), Qt.GlobalColor.black)

        if self.current_frame is not None:
            scaled_img = self.current_frame.scaled(
                self.ui.openGLWidget.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.ui.openGLWidget.width() - scaled_img.width()) // 2
            y = (self.ui.openGLWidget.height() - scaled_img.height()) // 2
            painter.drawImage(x, y, scaled_img)
        painter.end()

    @pyqtSlot(bool, str)
    def on_model_loaded(self, success, message):
        """模型加载完成回调"""
        if success:
            self.ui.statusbar.showMessage(f"模型加载成功: {message}", 3000)
        else:
            QMessageBox.warning(self, "模型加载失败", message)
            self.ui.statusbar.showMessage(f"模型加载失败: {message}", 5000)

    @pyqtSlot(np.ndarray)
    def update_display(self, cv_img):
        if cv_img is None:
            return
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.current_frame = qt_image.copy()
        self.ui.openGLWidget.update()

    # --- 业务逻辑 ---

    def on_model_change(self):
        """当下拉框改变时，拼接完整路径并加载"""
        filename = self.ui.comboBox_select_model.currentText()

        # 简单的校验，防止加载空字符串或错误提示文字
        if not filename.endswith(".pt"):
            return

        # 拼接完整路径 D:\fly\fly_dl\model\weights\yolov10n.pt
        full_model_path = os.path.join(self.weights_dir, filename)

        if os.path.exists(full_model_path):
            print(f"切换模型: {full_model_path}")
            # 使用异步加载，避免界面卡顿
            self.detector.load_model_async(full_model_path)
        else:
            print(f"文件丢失: {full_model_path}")

    def on_conf_change(self):
        val = self.ui.horizontalSlider_adjust_confidence.value()
        conf = val / 100.0
        self.ui.label_confidence_num.setText(f"{conf:.2f}")
        self.detector.update_config(conf)

    def on_tracking_state_changed(self, state):
        """跟踪复选框状态变化处理"""
        # 如果正在运行检测，则重新启动以应用新模式
        if self.detector.isRunning():
            # 获取当前源
            source = getattr(self.detector, "source", 0)
            # 停止当前检测
            self.stop_detection()
            # 重新启动检测
            self.start_detection(source)

    def start_detection(self, source):
        # 检查是否启用了跟踪功能
        if self.tracking_checkbox.isChecked():
            self.detector.start_tracking(source)
        else:
            self.detector.start_video(source)

    def stop_detection(self):
        self.detector.stop()

    def select_video_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if file_name:
            self.video_path = file_name
            self.ui.statusbar.showMessage(f"已选择视频: {file_name}")

    def start_video_detection(self):
        if hasattr(self, "video_path"):
            self.start_detection(self.video_path)
        else:
            QMessageBox.warning(self, "提示", "请先选择视频文件！")

    def select_image_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.png *.jpeg)"
        )
        if file_name:
            self.image_path = file_name
            self.ui.statusbar.showMessage(f"已选择图片: {file_name}")
            # 预览原图
            img = cv2.imread(file_name)
            self.update_display(img)

    def detect_image(self):
        # 图片检测不支持跟踪功能，直接使用普通检测
        if hasattr(self, "image_path"):
            self.detector.detect_single_image(self.image_path)
        else:
            QMessageBox.warning(self, "提示", "请先选择图片文件！")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
