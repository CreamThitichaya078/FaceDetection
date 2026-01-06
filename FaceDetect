import face_recognition
import cv2

# โหลดรูปภาพ
overlay_image = cv2.imread('logo4.png', cv2.IMREAD_UNCHANGED) #โหลดทั้ง RGB + Alpha Channel

if overlay_image is None:
    print("Error: Could not load the overlay image. Check the file path and name.")
    exit()

# เปิดใช้งานกล้องเว็บแคม
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW) #เปิดกล้องเว็บแคม (index = 0 คือกล้องตัวแรก) / cv2.CAP_DSHOW ป้องกัน warning ของ Windows
cap.set(cv2.CAP_PROP_FPS, 60)

#เต็มจอ
# cv2.namedWindow('Face Recognition Live', cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty('Face Recognition Live', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960 )

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam is active. Press 'q' to quit.")

# ตัวแปรสำหรับควบคุมการประมวลผลเฟรม
process_this_frame = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #พลิกภาพ
    frame = cv2.flip(frame, 1)

    # การลดขนาดเพื่อประสิทธิภาพ
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) #ย่อเฟรมลงเหลือ 50% → ทำให้หาหน้าได้เร็วขึ้น
    rgb_small_frame = small_frame[:, :, ::-1] #แปลงจาก BGR (OpenCV) → RGB(ที่ face_recognition ต้องใช้)

    #ตรวจหาหน้า
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        print("Detected:", face_locations)
        #คืนค่าเป็น list ของตำแหน่งใบหน้า (top, right, bottom, left)
    process_this_frame = not process_this_frame #ทำให้ตรวจหาหน้า เฟรมเว้นเฟรม → ประหยัด CPU

    # วาดกรอบรอบหน้า
    for (top, right, bottom, left) in face_locations:
        scale = 1 / 0.5  # ≈ 3.03
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        #การคำนวณขนาดโลโก้ ทำให้โลโก้ไม่บิดเบี้ยว
        scale_logo = 1  # ย่อโลโก้เหลือ ...% ของความกว้างใบหน้า

        logo_width = int((right - left) * scale_logo)
        logo_height = int(logo_width * (overlay_image.shape[0] / overlay_image.shape[1]))

        if logo_width <= 0 or logo_height <= 0:
            continue
        # ปรับขนาด
        resized_logo = cv2.resize(overlay_image, (logo_width, logo_height))

        #กำหนดตำแหน่งวางโลโก้
        x_offset = left
        y_offset = top - logo_height - 50

        if y_offset < 0:
            y_offset = 0

        # กันเหนียว Index Out of Bounds
        y1, y2 = y_offset, y_offset + resized_logo.shape[0]
        x1, x2 = x_offset, x_offset + resized_logo.shape[1]

        if y2 > frame.shape[0] or x2 > frame.shape[1]:
            continue

        # ผสมโลโก้เข้ากับเฟรม
        if resized_logo.shape[2] == 4:
            # มี Alpha Channel
            alpha_s = resized_logo[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * resized_logo[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
        else:
            # ไม่มี Alpha Channel
            frame[y1:y2, x1:x2] = resized_logo

    cv2.imshow('Face Recognition Live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
