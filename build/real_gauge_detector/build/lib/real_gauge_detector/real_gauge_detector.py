#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
import math

class RealGaugeDetector(Node):
    def __init__(self):
        super().__init__('real_gauge_detector')
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict)
        
        self.is_scanning = True
        self.current_marker_id = None
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error("Camra not found!")
            return
            
        self.get_logger().info("Detector started")
        self.timer = self.create_timer(0.03, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        try:
            if self.is_scanning:
                frame = self.scan_for_markers(frame)
            else:
                frame = self.real_find_gauge(frame)
            
            cv2.imshow('Real Gauge Detector', frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")
            self.is_scanning = True

    def scan_for_markers(self, frame):
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        
        cv2.putText(frame, "Scanning", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        if ids is not None:
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id[0])
                self.get_logger().info(f"Finded ID: {marker_id}")
                self.is_scanning = False
                self.current_marker_id = marker_id
                
                frame = aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.putText(frame, "Finded! Scanning manometr", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break
        
        return frame

    def real_find_gauge(self, frame):
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            if self.current_marker_id in [int(id[0]) for id in ids]:
                for i, marker_id in enumerate(ids):
                    if int(marker_id[0]) == self.current_marker_id:
                        corner = corners[i][0]
                        x_center = int(np.mean(corner[:, 0]))
                        y_center = int(np.mean(corner[:, 1]))
                        
                        # Ищем манометр и стрелку
                        gauge_data = self.detect_gauge_with_needle(frame, x_center, y_center)
                        
                        if gauge_data:
                            center, radius, needle_angle = gauge_data
                            # Вычисляем показания
                            reading = self.calculate_reading(needle_angle)
                            
                            # Рисуем результат
                            self.draw_gauge_result(frame, center, radius, needle_angle, reading)
                            cv2.putText(frame, f"value: {reading:.1f} bar", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(frame, "Manometr finded",
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "manometr not found", 
                                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        break
            else:
                self.is_scanning = True
                self.current_marker_id = None
        else:
            self.is_scanning = True
            self.current_marker_id = None
            
        return frame

    def detect_gauge_with_needle(self, frame, marker_x, marker_y):
        """Ищем манометр и стрелку"""
        try:
            # Область поиска справа от метки
            x = marker_x + 50
            y = marker_y - 75
            w, h = 150, 150
            
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return None
            
            roi = frame[y:y+h, x:x+w]
            
            # Ищем круги
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=50,
                param1=50,
                param2=30,
                minRadius=40,
                maxRadius=70
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                circle = circles[0][0]
                center_x, center_y, radius = circle
                
                # Координаты в основном изображении
                abs_center_x = x + center_x
                abs_center_y = y + center_y
                center = (abs_center_x, abs_center_y)
                
                # Ищем стрелку внутри круга
                needle_angle = self.detect_needle_in_circle(roi, center_x, center_y, radius)
                
                if needle_angle is not None:
                    return (center, radius, needle_angle)
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"error man finding: {str(e)}")
            return None

    def detect_needle_in_circle(self, roi, center_x, center_y, radius):
        """Ищем стрелку внутри круга"""
        try:
            # Создаем маску круга
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            
            # Ищем контрастные линии (стрелку)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Применяем маску круга
            edges = cv2.bitwise_and(edges, edges, mask=mask)
            
            # Ищем линии внутри круга
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=25, 
                                  minLineLength=20, maxLineGap=10)
            
            if lines is not None:
                # Находим самую длинную линию (стрелку)
                longest_line = None
                max_length = 0
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    
                    if length > max_length:
                        max_length = length
                        longest_line = (x1, y1, x2, y2)
                
                if longest_line:
                    x1, y1, x2, y2 = longest_line
                    # Вычисляем угол стрелки
                    angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
                    if angle < 0:
                        angle += 360
                    return angle
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"Ошибка поиска стрелки: {str(e)}")
            return None

    def calculate_reading(self, needle_angle):
        """Вычисляем показания на основе угла стрелки"""
        # Предполагаем шкалу 0-10 bar, стрелка движется 0-270 градусов
        min_angle = 0
        max_angle = 270
        min_value = 0
        max_value = 10
        
        # Нормализуем угол
        normalized_angle = max(min_angle, min(needle_angle, max_angle))
        
        # Линейное преобразование угла в значение
        reading = min_value + (normalized_angle / max_angle) * (max_value - min_value)
        return reading

    def draw_gauge_result(self, frame, center, radius, needle_angle, reading):
        """Рисуем манометр и стрелку"""
        # Рисуем круг манометра
        cv2.circle(frame, center, radius, (0, 255, 0), 3)
        cv2.circle(frame, center, 2, (0, 0, 255), 3)
        
        # Рисуем стрелку
        rad = np.radians(needle_angle)
        end_x = int(center[0] + (radius - 10) * np.cos(rad))
        end_y = int(center[1] + (radius - 10) * np.sin(rad))
        cv2.line(frame, center, (end_x, end_y), (0, 255, 0), 3)
        
        # Рисуем шкалу
        for angle in range(0, 271, 45):
            rad = np.radians(angle)
            start_x = int(center[0] + (radius - 5) * np.cos(rad))
            start_y = int(center[1] + (radius - 5) * np.sin(rad))
            end_x = int(center[0] + radius * np.cos(rad))
            end_y = int(center[1] + radius * np.sin(rad))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = RealGaugeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()