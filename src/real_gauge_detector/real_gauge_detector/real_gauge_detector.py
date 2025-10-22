#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import cv2.aruco as aruco
import numpy as np
import math
from collections import deque

class RealGaugeDetector(Node):
    def __init__(self):
        super().__init__('real_gauge_detector')
        
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dict)
        
        self.is_scanning = True
        self.current_marker_id = None
        self.last_gauge_center = None
        self.last_gauge_radius = None
        
        # Стабилизация стрелки
        self.needle_angles = deque(maxlen=5)
        self.stable_needle_angle = None
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.cap.isOpened():
            self.get_logger().error("Camera not found")
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
                frame = self.full_frame_gauge_detection(frame)
            
            cv2.imshow('Real Gauge Detector', frame)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Err: {str(e)}")
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
                cv2.putText(frame, "find manometr...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                break
        
        return frame

    def full_frame_gauge_detection(self, frame):
        corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
        
        if ids is not None:
            frame = aruco.drawDetectedMarkers(frame, corners, ids)
            
            if self.current_marker_id in [int(id[0]) for id in ids]:
                gauge_data = self.find_gauge_full_frame(frame)
                
                if gauge_data:
                    center, radius = gauge_data
                    self.last_gauge_center = center
                    self.last_gauge_radius = radius
                    
                    # Рисуем манометр
                    cv2.circle(frame, center, radius, (0, 255, 0), 3)
                    cv2.circle(frame, center, 2, (0, 0, 255), 3)
                    cv2.putText(frame, f"R={radius}", 
                               (center[0]-40, center[1]-radius-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Ищем стрелку с ПРАВИЛЬНЫМ определением направления
                    needle_angle = self.find_needle_correct_direction(frame, center, radius)
                    
                    if needle_angle is not None:
                        reading = self.calculate_reading(needle_angle)
                        
                        cv2.putText(frame, f"value: {reading:.1f} bar", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.putText(frame, f"corner: {needle_angle:.1f}°", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        
                        # Рисуем стрелку
                        rad = np.radians(needle_angle)
                        end_x = int(center[0] + (radius - 10) * np.cos(rad))
                        end_y = int(center[1] + (radius - 10) * np.sin(rad))
                        cv2.line(frame, center, (end_x, end_y), (0, 0, 255), 3)
                        
                        # Рисуем шкалу
                        self.draw_scale(frame, center, radius)
                    
                    cv2.putText(frame, "manometr finded", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    if self.last_gauge_center and self.last_gauge_radius:
                        center, radius = self.last_gauge_center, self.last_gauge_radius
                        cv2.circle(frame, center, radius, (0, 165, 255), 2)
                        cv2.putText(frame, "lost", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        cv2.putText(frame, "manometr not found", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                self.is_scanning = True
                self.current_marker_id = None
                self.last_gauge_center = None
        else:
            self.is_scanning = True
            self.current_marker_id = None
            self.last_gauge_center = None
            
        return frame

    def find_gauge_full_frame(self, frame):
        """Поиск манометра во всем кадре"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            
            circles = cv2.HoughCircles(
                gray, 
                cv2.HOUGH_GRADIENT, 
                dp=1, 
                minDist=100,
                param1=80,
                param2=30,
                minRadius=30,
                maxRadius=100
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles))
                
                valid_circles = []
                for circle in circles[0, :]:
                    center, radius = (circle[0], circle[1]), circle[2]
                    
                    if 35 <= radius <= 80:
                        if (radius < center[0] < frame.shape[1] - radius and 
                            radius < center[1] < frame.shape[0] - radius):
                            valid_circles.append((center, radius))
                
                if valid_circles:
                    if self.last_gauge_center:
                        best_circle = min(valid_circles, 
                                        key=lambda c: np.sqrt((c[0][0]-self.last_gauge_center[0])**2 + 
                                                            (c[0][1]-self.last_gauge_center[1])**2))
                        distance = np.sqrt((best_circle[0][0]-self.last_gauge_center[0])**2 + 
                                         (best_circle[0][1]-self.last_gauge_center[1])**2)
                        if distance < 50:
                            return best_circle
                    
                    return valid_circles[0]
            
            return None
            
        except Exception as e:
            self.get_logger().error(f"find error: {str(e)}")
            return None

    def find_needle_correct_direction(self, frame, center, radius):
        """ПРАВИЛЬНОЕ определение направления стрелки"""
        try:
            x = max(0, center[0] - radius)
            y = max(0, center[1] - radius)
            w = min(radius * 2, frame.shape[1] - x)
            h = min(radius * 2, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
                
            roi = frame[y:y+h, x:x+w]
            roi_center_x = center[0] - x
            roi_center_y = center[1] - y
            
            # Поиск КОНТУРОВ вместо линий - это ключевое изменение!
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Адаптивный порог для лучшего выделения стрелки
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Маска круга
            mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (roi_center_x, roi_center_y), int(radius*0.9), 255, -1)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
            
            # Морфологические операции для очистки
            kernel = np.ones((3,3), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Находим контуры
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            current_angle = None
            if contours:
                # Фильтруем контуры по площади и форме
                needle_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 500:  # Подходящий размер для стрелки
                        # Проверяем вытянутость
                        rect = cv2.minAreaRect(contour)
                        width, height = rect[1]
                        if min(width, height) > 0:
                            aspect_ratio = max(width, height) / min(width, height)
                            if aspect_ratio > 2.0:  # Вытянутый объект
                                needle_contours.append(contour)
                
                if needle_contours:
                    # Берем самый большой вытянутый контур
                    needle_contour = max(needle_contours, key=cv2.contourArea)
                    
                    # НОВЫЙ СПОСОБ: определяем направление по геометрии контура
                    
                    # 1. Находим ось контура
                    [vx, vy, x, y] = cv2.fitLine(needle_contour, cv2.DIST_L2, 0, 0.01, 0.01)
                    line_angle = math.atan2(vy, vx) * 180 / math.pi
                    
                    # 2. Находим самую удаленную точку контура от центра
                    max_dist = 0
                    farthest_point = None
                    
                    for point in needle_contour:
                        px, py = point[0]
                        dist = np.sqrt((px - roi_center_x)**2 + (py - roi_center_y)**2)
                        if dist > max_dist:
                            max_dist = dist
                            farthest_point = (px, py)
                    
                    if farthest_point:
                        # 3. Вычисляем угол до самой удаленной точки
                        target_angle = math.atan2(farthest_point[1] - roi_center_y, 
                                                farthest_point[0] - roi_center_x) * 180 / math.pi
                        
                        # 4. Сравниваем с углом линии
                        angle_diff = abs(line_angle - target_angle)
                        
                        # Если разница небольшая, используем угол до удаленной точки
                        if angle_diff < 45 or angle_diff > 135:
                            current_angle = target_angle
                        else:
                            # Иначе используем угол линии, но корректируем направление
                            current_angle = line_angle
                            
                            # Проверяем какое направление правильное
                            test_x = roi_center_x + 20 * np.cos(np.radians(line_angle))
                            test_y = roi_center_y + 20 * np.sin(np.radians(line_angle))
                            test_dist = np.sqrt((test_x - roi_center_x)**2 + (test_y - roi_center_y)**2)
                            
                            opposite_angle = (line_angle + 180) % 360
                            test_x2 = roi_center_x + 20 * np.cos(np.radians(opposite_angle))
                            test_y2 = roi_center_y + 20 * np.sin(np.radians(opposite_angle))
                            test_dist2 = np.sqrt((test_x2 - roi_center_x)**2 + (test_y2 - roi_center_y)**2)
                            
                            # Выбираем направление, которое больше соответствует удаленной точке
                            if abs(test_angle - target_angle) > abs(opposite_angle - target_angle):
                                current_angle = opposite_angle
                    
                    if current_angle is not None:
                        if current_angle < 0:
                            current_angle += 360
            
            # Стабилизация
            if current_angle is not None:
                self.needle_angles.append(current_angle)
                
                if len(self.needle_angles) >= 3:
                    stable_angle = np.median(list(self.needle_angles))
                    
                    if self.stable_needle_angle is not None:
                        angle_diff = abs(stable_angle - self.stable_needle_angle)
                        if angle_diff < 30:
                            self.stable_needle_angle = stable_angle
                    else:
                        self.stable_needle_angle = stable_angle
                else:
                    self.stable_needle_angle = current_angle
            
            return self.stable_needle_angle
            
        except Exception as e:
            self.get_logger().error(f"error find arrow: {str(e)}")
            return self.stable_needle_angle

    def calculate_reading(self, needle_angle):
        """Вычисляем показания"""
        normalized_angle = needle_angle % 360
        
        if normalized_angle > 180:
            reading = ((360 - normalized_angle) / 180) * 10
        else:
            reading = 10 - (normalized_angle / 180) * 10
        
        return max(0, min(reading, 10))

    def draw_scale(self, frame, center, radius):
        """Рисуем шкалу"""
        for angle in range(0, 181, 30):
            rad = np.radians(angle)
            start_x = int(center[0] + (radius - 8) * np.cos(rad))
            start_y = int(center[1] + (radius - 8) * np.sin(rad))
            end_x = int(center[0] + radius * np.cos(rad))
            end_y = int(center[1] + radius * np.sin(rad))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)
            
            value = 10 - (angle / 180) * 10
            text_x = int(center[0] + (radius + 15) * np.cos(rad))
            text_y = int(center[1] + (radius + 15) * np.sin(rad))
            cv2.putText(frame, f"{value:.0f}", (text_x-8, text_y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

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