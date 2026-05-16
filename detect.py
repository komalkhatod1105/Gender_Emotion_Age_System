# =========================================================
# STABLE AI DETECTION SYSTEM
# FIXED CAMERA VERSION
# =========================================================

import cv2
from deepface import DeepFace
import time
from datetime import datetime

# =========================================================
# CAMERA SETUP
# =========================================================

# FIXED CAMERA BACKEND FOR WINDOWS
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Resolution
cap.set(3, 1280)
cap.set(4, 720)

# Check Camera
if not cap.isOpened():
    print("Camera not detected")
    exit()

# =========================================================
# VARIABLES
# =========================================================

prev_time = 0
frame_count = 0
photo_count = 0

last_gender = "Detecting..."
last_age_group = "Unknown"
last_emotion = "Neutral"

face_data = []

print("======================================")
print(" AI DETECTION SYSTEM STARTED ")
print("======================================")
print("ESC = Exit")
print("S   = Save Screenshot")
print("======================================")

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Mirror Effect
    frame = cv2.flip(frame, 1)

    # Resize
    frame = cv2.resize(frame, (960, 720))

    current_time = time.time()

    frame_count += 1

    # =========================================================
    # ANALYZE EVERY 10 FRAMES
    # =========================================================

    if frame_count % 10 == 0:

        try:

            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                detector_backend='retinaface',
                enforce_detection=False,
                silent=True
            )

            if not isinstance(results, list):
                results = [results]

            face_data = []

            # =========================================================
            # PROCESS EACH FACE
            # =========================================================

            for result in results:

                x = result['region']['x']
                y = result['region']['y']
                w = result['region']['w']
                h = result['region']['h']

                # =========================================================
                # BETTER GENDER DETECTION
                # =========================================================

                male_score = result['gender']['Man']
                female_score = result['gender']['Woman']

                if male_score > female_score:
                    gender = "Male"
                else:
                    gender = "Female"

                # =========================================================
                # BETTER AGE GROUP
                # =========================================================

                age = int(result['age'])

                if age < 13:
                    age_group = "Child"

                elif age < 20:
                    age_group = "Teen"

                elif age < 35:
                    age_group = "Young Adult"

                elif age < 50:
                    age_group = "Adult"

                else:
                    age_group = "Senior"

                # =========================================================
                # EMOTION
                # =========================================================

                emotion = result['dominant_emotion']

                # =========================================================
                # COLORS
                # =========================================================

                if emotion == "happy":
                    color = (0, 255, 255)

                elif emotion == "sad":
                    color = (255, 0, 0)

                elif emotion == "angry":
                    color = (0, 0, 255)

                else:
                    color = (0, 255, 0)

                # Store face data
                face_data.append({
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "gender": gender,
                    "age_group": age_group,
                    "emotion": emotion,
                    "color": color
                })

        except Exception as e:
            print("Detection Error:", e)

    # =========================================================
    # DRAW FACE DATA
    # =========================================================

    for face in face_data:

        x = face["x"]
        y = face["y"]
        w = face["w"]
        h = face["h"]

        gender = face["gender"]
        age_group = face["age_group"]
        emotion = face["emotion"]
        color = face["color"]

        # Face Box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        # Gender
        cv2.putText(
            frame,
            f"Gender : {gender}",
            (x, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Age Group
        cv2.putText(
            frame,
            f"Age : {age_group}",
            (x, y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Emotion
        cv2.putText(
            frame,
            f"Emotion : {emotion}",
            (x, y + h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    # =========================================================
    # FPS
    # =========================================================

    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # =========================================================
    # TOP PANEL
    # =========================================================

    cv2.rectangle(frame, (0, 0), (420, 120), (0, 0, 0), -1)

    # FPS
    cv2.putText(
        frame,
        f"FPS : {int(fps)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Time
    current_clock = datetime.now().strftime("%H:%M:%S")

    cv2.putText(
        frame,
        current_clock,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Date
    current_date = datetime.now().strftime("%d-%m-%Y")

    cv2.putText(
        frame,
        current_date,
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # =========================================================
    # BOTTOM PANEL
    # =========================================================

    cv2.rectangle(frame, (0, 680), (960, 720), (0, 0, 0), -1)

    cv2.putText(
        frame,
        "Press S = Screenshot | ESC = Exit",
        (20, 708),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # =========================================================
    # SHOW WINDOW
    # =========================================================

    cv2.imshow("Stable AI Detection System", frame)

    # =========================================================
    # KEYBOARD
    # =========================================================

    key = cv2.waitKey(1)

    # ESC Exit
    if key == 27:
        break

    # Screenshot
    elif key == ord('s'):

        filename = f"screenshot_{photo_count}.jpg"

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        photo_count += 1

# =========================================================
# RELEASE
# =========================================================

cap.release()
cv2.destroyAllWindows()