import cv2
import sys

img = cv2.imread("../data/calibration/WhatsApp Image 2026-10-03 at 9.49.24 PM (1).jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

found = False
for w in range(15, 25):
    for h in range(5, 9):
        ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
        if ret:
            print(f"FOUND grid size: ({w}, {h})")
            found = True
            break
    if found: break

if not found:
    print("Could not find any grid between 15x5 and 25x8")
