import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
import pygame
import time
import keyboard

face_cascade = cv.CascadeClassifier(r'C:\Users\Lenovo\Desktop\DRIVE_DROWSINESS\model_cascad\haarcascade_frontalface_alt.xml')
left_eye_cascade = cv.CascadeClassifier(r'C:\Users\Lenovo\Desktop\DRIVE_DROWSINESS\model_cascad\haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv.CascadeClassifier(r'C:\Users\Lenovo\Desktop\DRIVE_DROWSINESS\model_cascad\haarcascade_righteye_2splits.xml')
model = load_model(r"C:\Users\Lenovo\Desktop\DRIVE_DROWSINESS\model VGG\VggNet.keras")

pygame.mixer.init()
audio_alert_path = r'C:\Users\Lenovo\Desktop\DRIVE_DROWSINESS\code\audio_alert.wav'

def prepare_image(image):
    img_resized = cv.resize(image, (64, 64))
    img_normalized = img_resized / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded

eyes_closed_start = None
alert_playing = False

cap = cv.VideoCapture(0)

while cap.isOpened():
    if keyboard.is_pressed("ctrl+z"):
        print("Fin de la détection (Ctrl+Z pressé).")
        break

    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire depuis la webcam.")
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    result_general = None
    left_prediction = None
    right_prediction = None

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        left_eye = left_eye_cascade.detectMultiScale(roi_gray)
        for (lx, ly, lw, lh) in left_eye:
            l_eye = roi_color[ly:ly + lh, lx:lx + lw]
            preprocessed_eye = prepare_image(l_eye)
            left_prediction = model.predict(preprocessed_eye)[0][0]
            color = (0, 255, 0) if left_prediction > 0.8 else (0, 0, 255)
            cv.rectangle(roi_color, (lx, ly), (lx + lw, ly + lh), color, 2)
            break

        right_eye = right_eye_cascade.detectMultiScale(roi_gray)
        for (rx, ry, rw, rh) in right_eye:
            r_eye = roi_color[ry:ry + rh, rx:rx + rw]
            preprocessed_eye = prepare_image(r_eye)
            right_prediction = model.predict(preprocessed_eye)[0][0]
            color = (0, 255, 0) if right_prediction > 0.8 else (0, 0, 255)
            cv.rectangle(roi_color, (rx, ry), (rx + rw, ry + rh), color, 2)
            break

    if left_prediction is not None and right_prediction is not None:
        if left_prediction > 0.8 and right_prediction > 0.8:
            result_general = "Open EYES"
            color_general = (0, 255, 0)
            eyes_closed_start = None
            if alert_playing:
                pygame.mixer.music.stop()
                alert_playing = False
        elif left_prediction <= 0.8 and right_prediction <= 0.8:
            result_general = "Close EYES"
            color_general = (0, 0, 255)

            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            elif time.time() - eyes_closed_start > 2:
                if not alert_playing:
                    pygame.mixer.music.load(audio_alert_path)
                    pygame.mixer.music.play(-1)  # Jouer en boucle
                    alert_playing = True
        else:
            result_general = "HAHA"
            color_general = (0, 255, 255)

        cv.putText(frame, result_general, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color_general, 2)

    cv.imshow("Eye State Detection - Real Time", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Fin de la détection (Q pressé).")
        break

cap.release()
cv.destroyAllWindows()
pygame.mixer.quit()