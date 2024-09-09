import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
import numpy as np

# Inisialisasi model tangan dari mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Layar untuk membaca gestur
browser_opened = False
screen_width, screen_height = pyautogui.size()

# Variabel untuk membaca kepalan tangan
fist_timer = None  # Timer untuk membaca kepalan tangan
fist_duration = 5  # Durasi untuk membaca kepalan tangan(per detik)

# Menyimpan posisi kursor sebelumnya untuk interpolasi smooth
prev_cursor_x, prev_cursor_y = screen_width // 2, screen_height // 2
smooth_factor = 0.2  # Faktor untuk mengontrol kelancaran

# Fungsi untuk membuka browser
def open_browser():
    global browser_opened
    if not browser_opened:
        webbrowser.open("https://www.youtube.com")
        time.sleep(3)  # Beri waktu untuk membuka browser
        browser_opened = True

# Scroll halaman web berdasarkan posisi kursor di atas/bawah layar
def auto_scroll(cursor_y):
    scroll_threshold_top = 50  # Jarak dari bagian atas layar
    scroll_threshold_bottom = screen_height - 50  # Jarak dari bagian bawah layar
    scroll_amount = 800  # Ukuran pergerakan scroll

    # Jika kursor berada dekat bagian atas, scroll ke atas
    if cursor_y <= scroll_threshold_top:
        pyautogui.scroll(scroll_amount)
        print("Scrolling up")

    # Jika kursor berada dekat bagian bawah, scroll ke bawah
    elif cursor_y >= scroll_threshold_bottom:
        pyautogui.scroll(-scroll_amount)
        print("Scrolling down")

# Gerakan kursor yang lancar dengan interpolasi posisi
def smooth_cursor_move(cursor_x, cursor_y):
    global prev_cursor_x, prev_cursor_y
    # Interpolasi untuk membuat pergerakan lebih halus
    new_x = prev_cursor_x + (cursor_x - prev_cursor_x) * smooth_factor
    new_y = prev_cursor_y + (cursor_y - prev_cursor_y) * smooth_factor

    pyautogui.moveTo(new_x, new_y)
    prev_cursor_x, prev_cursor_y = new_x, new_y

    # Menjalankan fungsi auto-scroll ketika kursor berada di posisi atas atau bawah
    auto_scroll(new_y)

# Mengatur volume berdasarkan jarak kedua tangan
def adjust_volume(distance):
    if distance > 0.4:
        print("Increasing Volume")
        for _ in range(10):  # Menambah volume lebih banyak berdasarkan menjauhnya kedua tangan
            pyautogui.press('volumeup')  # Menjelaskan kondisi ketika volume ditambahkan
    elif distance < 0.2:
        print("Decreasing Volume")
        for _ in range(10):  # Mengurangi volume lebih banyak berdasarkan mendekatkan kedua tangan
            pyautogui.press('volumedown')  # Menjelaskan kondisi ketika volume dikurangi

# Mendeteksi apabila tangan pada kondisi mengepal
def detect_fist(hand_landmark):
    # Melakukan cek apabila semua jari tertutup
    thumb_tip = hand_landmark.landmark[4]
    index_tip = hand_landmark.landmark[8]
    middle_tip = hand_landmark.landmark[12]
    ring_tip = hand_landmark.landmark[16]
    pinky_tip = hand_landmark.landmark[20]
    
    if (index_tip.y > hand_landmark.landmark[6].y and
        middle_tip.y > hand_landmark.landmark[10].y and
        ring_tip.y > hand_landmark.landmark[14].y and
        pinky_tip.y > hand_landmark.landmark[18].y):
        return True
    return False

# Membaca gesture untuk bisa mengatur browser dan video
def handle_gestures(hand_landmarks):
    global browser_opened, fist_timer

    # Mendapatkan tangan kiri dan kanan(Apabila keduanya terdeteksi)
    right_hand = hand_landmarks[0]
    left_hand = hand_landmarks[1] if len(hand_landmarks) > 1 else None

    # Gestur: tangan kanan dengan semua jari terbuka(Mengatur kursor)
    if right_hand.landmark[8].y < right_hand.landmark[6].y and not browser_opened:
        print("Right hand raised - Opening YouTube")
        open_browser()

    # Mengatur tangan kanan untuk menggerakan kursor dan klik
    if right_hand:
        right_fingers_open = all(
            right_hand.landmark[i].y < right_hand.landmark[i - 2].y for i in [8, 12, 16, 20]
        )  # Semua tangan kanan terbuka

        # Kondisi untuk menggerakan kursor jika semua jari tangan kanan dibuka
        if right_fingers_open:
            cursor_x = np.interp(right_hand.landmark[8].x, [0, 1], [0, screen_width])
            cursor_y = np.interp(right_hand.landmark[8].y, [0, 1], [0, screen_height])
            smooth_cursor_move(cursor_x, cursor_y)

        # Kondisi untuk mengklik kursor jika hanya jari telunjuk yang terbuka pada tangan kanan
        if right_hand.landmark[8].y < right_hand.landmark[6].y and right_hand.landmark[12].y > right_hand.landmark[10].y:
            print("Right hand index finger only - Clicking at cursor position")
            pyautogui.click()

    # Kedua tangan terdeteksi (Mengatur volume)
    if right_hand and left_hand:
        # Mengatur jarak antara kedua tangan
        right_index_finger = np.array([right_hand.landmark[8].x, right_hand.landmark[8].y])
        left_index_finger = np.array([left_hand.landmark[8].x, left_hand.landmark[8].y])
        distance_between_hands = np.linalg.norm(right_index_finger - left_index_finger)

        # Mengatur volume berdasarkan jarak kedua tangan tersebut
        adjust_volume(distance_between_hands)

    # Mendeteksi kepalan tangan dan memulai timer
    if detect_fist(right_hand):
        if fist_timer is None:
            fist_timer = time.time()
        elif time.time() - fist_timer >= fist_duration:
            print("Fist detected for 5 seconds - Exiting the program")
            exit()
    else:
        fist_timer = None  # Mereset timer apabila kepalan tidak terdeteksi

# Fungsi utama untuk mendeteksi gestur dan membuka kamera
def main():
    cap = cv2.VideoCapture(0)  # Membuka kamera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Mengatur ukuran untuk layar kamera
        frame = cv2.resize(frame, (1100, 700))

        # Memutar frame kamera secara horizontal agar terdeteksi secara benar
        frame = cv2.flip(frame, 1)

        # Konversi warna gambar pada mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Memberikan gambaran pada deteksi tangan
        results = hands.process(image_rgb)

        # Menggambar tangan apabila tangan terdeteksi dilayar
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks

            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

            # Mendapatkan jumlah tangan yang terdeteksi
            if len(hand_landmarks) > 0:
                handle_gestures(hand_landmarks)

        cv2.imshow("Hand Gesture Control", frame)

        # Menutup program apabila tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Menutup kamera
    cap.release()
    cv2.destroyAllWindows()

# Menjalankan program utama
if __name__ == "__main__":
    main()
