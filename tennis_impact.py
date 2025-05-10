import cv2
import numpy as np
import time

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_COLOR = (0, 255, 0)

MIN_AREA = 500
MIN_RADIUS = 5
COOLDOWN = 15

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

physical_area = np.array([[100, 100], [1180, 100], [1180, 620], [100, 620]], dtype=np.float32)
virtual_area = np.array([[0, 0], [FRAME_WIDTH, 0], [FRAME_WIDTH, FRAME_HEIGHT], [0, FRAME_HEIGHT]], dtype=np.float32)

H, _ = cv2.findHomography(physical_area, virtual_area)
virtual_screen = np.ones((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) * 255

fgbg = cv2.createBackgroundSubtractorMOG2()
log = open("hit_log.csv", "w")
log.write("Frame,X,Y,Timestamp\n")

last_hit = -COOLDOWN
frame_count = 0

def mark_hit(x, y):
    x = int(np.clip(x, 0, FRAME_WIDTH - 1))
    y = int(np.clip(y, 0, FRAME_HEIGHT - 1))
    cv2.circle(virtual_screen, (x, y), 15, (0, 0, 0), -1)

def process(frame):
    global last_hit, frame_count

    mask = fgbg.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv2.contourArea(c) > MIN_AREA:
            (x, y), r = cv2.minEnclosingCircle(c)
            if r > MIN_RADIUS and frame_count - last_hit >= COOLDOWN:
                last_hit = frame_count
                pt = np.array([[[x, y]]], dtype=np.float32)
                mapped = cv2.perspectiveTransform(pt, H)
                vx, vy = mapped[0][0]
                mark_hit(vx, vy)
                log.write(f"{frame_count},{int(vx)},{int(vy)},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.flush()

print("Running. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    process(frame)
    cv2.polylines(frame, [physical_area.astype(int)], True, FRAME_COLOR, 3)
    cv2.imshow('Webcam View', frame)
    cv2.imshow('Virtual Screen', virtual_screen)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

log.close()
cap.release()
cv2.destroyAllWindows()
