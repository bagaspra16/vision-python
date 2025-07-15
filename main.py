import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
import numpy as np
import platform
import subprocess
import sys
from typing import Optional, Tuple, List

class CrossPlatformGestureControl:
    def __init__(self):
        self.system = platform.system()
        self.setup_pyautogui()
        self.setup_mediapipe()
        self.setup_screen_parameters()
        self.setup_gesture_variables()
        
    def setup_pyautogui(self):
        """Setup pyautogui dengan fail-safe untuk berbagai platform"""
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.1
        
        # Untuk macOS, nonaktifkan fail-safe di sudut kiri atas
        if self.system == "Darwin":
            pyautogui.FAILSAFE = False
            
    def setup_mediapipe(self):
        """Inisialisasi MediaPipe hands"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def setup_screen_parameters(self):
        """Setup parameter layar yang adaptif"""
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Adaptasi untuk berbagai resolusi
        if self.screen_width >= 2560:  # 4K atau lebih tinggi
            self.camera_width, self.camera_height = 1280, 720
            self.scroll_threshold = 200
            self.scroll_amount = 300
        elif self.screen_width >= 1920:  # Full HD
            self.camera_width, self.camera_height = 1024, 576
            self.scroll_threshold = 150
            self.scroll_amount = 200
        else:  # HD atau lebih rendah
            self.camera_width, self.camera_height = 800, 450
            self.scroll_threshold = 100
            self.scroll_amount = 150
            
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        print(f"Camera resolution: {self.camera_width}x{self.camera_height}")
        
    def setup_gesture_variables(self):
        """Setup variabel untuk gesture control"""
        self.browser_opened = False
        self.fist_timer = None
        self.fist_duration = 5
        self.prev_cursor_x = self.screen_width // 2
        self.prev_cursor_y = self.screen_height // 2
        self.smooth_factor = 0.2
        self.last_click_time = 0
        self.click_cooldown = 0.5  # Cooldown untuk menghindari multiple click
        
    def get_available_cameras(self) -> List[int]:
        """Mendapatkan daftar kamera yang tersedia"""
        available_cameras = []
        for i in range(5):  # Cek 5 kamera pertama
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def open_browser_cross_platform(self):
        """Membuka browser dengan cara yang kompatibel cross-platform"""
        if not self.browser_opened:
            try:
                if self.system == "Darwin":  # macOS
                    webbrowser.open("https://www.youtube.com")
                elif self.system == "Windows":
                    webbrowser.open("https://www.youtube.com")
                elif self.system == "Linux":
                    webbrowser.open("https://www.youtube.com")
                    
                time.sleep(3)
                self.browser_opened = True
                print("Browser opened successfully")
            except Exception as e:
                print(f"Error opening browser: {e}")
                
    def adjust_volume_cross_platform(self, distance: float):
        """Mengatur volume dengan cara yang kompatibel cross-platform"""
        try:
            if distance > 0.4:
                print("Increasing Volume")
                if self.system == "Darwin":  # macOS
                    for _ in range(5):
                        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
                elif self.system == "Windows":
                    for _ in range(5):
                        pyautogui.press('volumeup')
                elif self.system == "Linux":
                    for _ in range(5):
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%+"])
                        
            elif distance < 0.2:
                print("Decreasing Volume")
                if self.system == "Darwin":  # macOS
                    for _ in range(5):
                        subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
                elif self.system == "Windows":
                    for _ in range(5):
                        pyautogui.press('volumedown')
                elif self.system == "Linux":
                    for _ in range(5):
                        subprocess.run(["amixer", "-D", "pulse", "sset", "Master", "5%-"])
                        
        except Exception as e:
            print(f"Error adjusting volume: {e}")
            
    def auto_scroll(self, cursor_y: float):
        """Auto scroll berdasarkan posisi kursor dengan adaptasi layar"""
        scroll_threshold_top = self.scroll_threshold
        scroll_threshold_bottom = self.screen_height - self.scroll_threshold
        
        if cursor_y <= scroll_threshold_top:
            print("Scrolling up")
            try:
                pyautogui.scroll(self.scroll_amount)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error scrolling up: {e}")
                
        elif cursor_y >= scroll_threshold_bottom:
            print("Scrolling down")
            try:
                pyautogui.scroll(-self.scroll_amount)
                time.sleep(0.1)
            except Exception as e:
                print(f"Error scrolling down: {e}")
                
    def smooth_cursor_move(self, cursor_x: float, cursor_y: float):
        """Gerakan kursor yang halus dengan interpolasi"""
        try:
            new_x = self.prev_cursor_x + (cursor_x - self.prev_cursor_x) * self.smooth_factor
            new_y = self.prev_cursor_y + (cursor_y - self.prev_cursor_y) * self.smooth_factor
            
            # Batasi gerakan dalam bounds layar
            new_x = max(0, min(new_x, self.screen_width - 1))
            new_y = max(0, min(new_y, self.screen_height - 1))
            
            pyautogui.moveTo(new_x, new_y)
            self.prev_cursor_x, self.prev_cursor_y = new_x, new_y
            
            self.auto_scroll(new_y)
            
        except Exception as e:
            print(f"Error moving cursor: {e}")
            
    def detect_fist(self, hand_landmark) -> bool:
        """Mendeteksi kepalan tangan yang lebih akurat"""
        try:
            # Cek apakah semua jari tertutup
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
            finger_pips = [6, 10, 14, 18]  # PIP joints
            
            fingers_closed = 0
            for tip, pip in zip(finger_tips, finger_pips):
                if hand_landmark.landmark[tip].y > hand_landmark.landmark[pip].y:
                    fingers_closed += 1
                    
            # Cek thumb (lebih kompleks karena bergerak horizontal)
            thumb_tip = hand_landmark.landmark[4]
            thumb_ip = hand_landmark.landmark[3]
            if thumb_tip.x > thumb_ip.x:  # Thumb closed (untuk right hand)
                fingers_closed += 1
                
            return fingers_closed >= 4
            
        except Exception as e:
            print(f"Error detecting fist: {e}")
            return False
            
    def handle_gestures(self, hand_landmarks):
        """Menangani berbagai gesture dengan error handling"""
        try:
            right_hand = hand_landmarks[0]
            left_hand = hand_landmarks[1] if len(hand_landmarks) > 1 else None
            
            # Gesture: buka browser dengan tangan kanan terbuka
            if not self.browser_opened:
                fingers_open = sum(1 for i in [8, 12, 16, 20] 
                                 if right_hand.landmark[i].y < right_hand.landmark[i-2].y)
                if fingers_open >= 3:
                    print("Opening browser...")
                    self.open_browser_cross_platform()
                    
            # Kontrol kursor dengan tangan kanan
            if right_hand:
                right_fingers_open = all(
                    right_hand.landmark[i].y < right_hand.landmark[i-2].y 
                    for i in [8, 12, 16, 20]
                )
                
                if right_fingers_open:
                    cursor_x = np.interp(right_hand.landmark[8].x, [0, 1], [0, self.screen_width])
                    cursor_y = np.interp(right_hand.landmark[8].y, [0, 1], [0, self.screen_height])
                    self.smooth_cursor_move(cursor_x, cursor_y)
                    
                # Click dengan hanya jari telunjuk terbuka
                current_time = time.time()
                if (right_hand.landmark[8].y < right_hand.landmark[6].y and 
                    right_hand.landmark[12].y > right_hand.landmark[10].y and
                    current_time - self.last_click_time > self.click_cooldown):
                    print("Clicking...")
                    pyautogui.click()
                    self.last_click_time = current_time
                    
            # Kontrol volume dengan kedua tangan
            if right_hand and left_hand:
                right_index = np.array([right_hand.landmark[8].x, right_hand.landmark[8].y])
                left_index = np.array([left_hand.landmark[8].x, left_hand.landmark[8].y])
                distance = np.linalg.norm(right_index - left_index)
                self.adjust_volume_cross_platform(distance)
                
            # Deteksi kepalan untuk keluar
            if self.detect_fist(right_hand):
                if self.fist_timer is None:
                    self.fist_timer = time.time()
                elif time.time() - self.fist_timer >= self.fist_duration:
                    print("Fist detected for 5 seconds - Exiting...")
                    return False
            else:
                self.fist_timer = None
                
        except Exception as e:
            print(f"Error handling gestures: {e}")
            
        return True
        
    def run(self):
        """Fungsi utama untuk menjalankan gesture control"""
        print(f"Running on {self.system}")
        print("Available cameras:", self.get_available_cameras())
        
        # Coba berbagai kamera
        cap = None
        for camera_id in self.get_available_cameras():
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                print(f"Using camera {camera_id}")
                break
                
        if not cap or not cap.isOpened():
            print("Error: Could not open any camera.")
            return
            
        # Setup kamera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Controls:")
        print("- Open all fingers on right hand: Open browser")
        print("- All fingers open: Move cursor")
        print("- Only index finger: Click")
        print("- Two hands distance: Control volume")
        print("- Make fist for 5 seconds: Exit")
        print("- Press 'q' to quit")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                    
                # Resize frame
                frame = cv2.resize(frame, (self.camera_width, self.camera_height))
                frame = cv2.flip(frame, 1)
                
                # Process dengan MediaPipe
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(image_rgb)
                
                # Draw landmarks
                if results.multi_hand_landmarks:
                    for hand_landmark in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmark, self.mp_hands.HAND_CONNECTIONS
                        )
                        
                    # Handle gestures
                    if not self.handle_gestures(results.multi_hand_landmarks):
                        break
                        
                # Show frame
                cv2.imshow("Cross-Platform Hand Gesture Control", frame)
                
                # Exit dengan 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hands.close()
            
def main():
    """Entry point"""
    try:
        controller = CrossPlatformGestureControl()
        controller.run()
    except Exception as e:
        print(f"Error initializing gesture control: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()