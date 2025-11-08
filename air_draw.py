"""
air_draw.py
Simple Air Drawing app using Mediapipe + OpenCV.

Usage:
    python air_draw.py
Press ESC to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ---------- CONFIG ----------
CAM_WIDTH = 1280
CAM_HEIGHT = 720

# Palette: list of (B,G,R) colors
PALETTE = [
    (255, 0, 0),    # blue
    (0, 255, 0),    # green
    (0, 0, 255),    # red
    (0, 255, 255),  # yellow (BGR)
    (255, 0, 255),  # magenta
    (255, 255, 255) # white/eraser
]

RECT_W = 110
RECT_H = 90
RECT_GAP = 10
TOP_PADDING = 10
LEFT_PADDING = 10

# ---------- Mediapipe init ----------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ---------- Helper functions ----------
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_palette(frame, selected_idx):
    x = LEFT_PADDING
    y = TOP_PADDING
    for i, col in enumerate(PALETTE):
        # rectangle
        cv2.rectangle(frame, (x, y), (x+RECT_W, y+RECT_H), col, -1)
        # border
        border_color = (0,0,0)
        thickness = 4 if i == selected_idx else 2
        cv2.rectangle(frame, (x, y), (x+RECT_W, y+RECT_H), border_color, thickness)
        # label
        label = "Eraser" if col == (255,255,255) else f"Color {i+1}"
        cv2.putText(frame, label, (x+8, y+RECT_H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        x += RECT_W + RECT_GAP

    # Clear button on rightmost
    clear_x = LEFT_PADDING + (RECT_W + RECT_GAP) * len(PALETTE)
    cv2.rectangle(frame, (clear_x, y), (clear_x+RECT_W, y+RECT_H), (50,50,50), -1)
    cv2.rectangle(frame, (clear_x, y), (clear_x+RECT_W, y+RECT_H), (0,0,0), 2)
    cv2.putText(frame, "Clear", (clear_x+18, y+RECT_H-30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    return clear_x, clear_x+RECT_W

# ---------- Main ----------
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)  # drawing canvas
    prev_x, prev_y = None, None
    selected_color_idx = 0
    drawing = False
    last_time = time.time()
    save_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not available")
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # draw palette UI
        clear_left, clear_right = draw_palette(frame, selected_color_idx)

        if results.multi_hand_landmarks:
            handLms = results.multi_hand_landmarks[0]
            lm = handLms.landmark

            h, w, _ = frame.shape
            # Coordinates of index tip (id 8) and middle tip (id 12)
            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            mx, my = int(lm[12].x * w), int(lm[12].y * h)

            # visual circles on finger tips
            cv2.circle(frame, (ix, iy), 8, (0, 255, 255), -1)
            cv2.circle(frame, (mx, my), 8, (0, 255, 255), -1)

            # distance between index and middle finger
            d = distance((ix, iy), (mx, my))

            # If index and middle are close -> selection mode (like pinch)
            if d < 40:
                drawing = False
                prev_x, prev_y = None, None
                # Check if index tip is in palette area (y within top rectangles)
                if iy < TOP_PADDING + RECT_H + 5:
                    # compute which box
                    x_rel = ix - LEFT_PADDING
                    if x_rel >= 0:
                        idx = x_rel // (RECT_W + RECT_GAP)
                        # check within last partial gap
                        if idx < len(PALETTE):
                            # check exact rectangle bounds (avoid gaps)
                            box_x_start = LEFT_PADDING + idx*(RECT_W + RECT_GAP)
                            box_x_end = box_x_start + RECT_W
                            if box_x_start < ix < box_x_end:
                                selected_color_idx = int(idx)
                                # visual feedback
                                cv2.putText(frame, f"Selected: {selected_color_idx+1}", (10, TOP_PADDING + RECT_H + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    # check clear button
                    if clear_left < ix < clear_right:
                        canvas[:] = 0
                        # small pause to show cleared
                        cv2.putText(frame, "Cleared", (CAM_WIDTH//2 - 80, CAM_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
                        cv2.imshow("AirDraw", frame)
                        cv2.waitKey(150)
                # if pinch elsewhere, do nothing except selection feedback
                cv2.line(frame, (ix, iy), (mx, my), (255,255,255), 2)
            else:
                # Drawing mode — index finger tip draws
                drawing = True
                color = PALETTE[selected_color_idx]
                # if eraser selected (white), draw thicker line with black canvas clearing
                if color == (255,255,255):
                    thickness = 40
                    draw_color = (0,0,0)  # draw black on canvas to "erase"
                else:
                    thickness = 10
                    draw_color = color

                if prev_x is None:
                    prev_x, prev_y = ix, iy

                # draw on canvas
                cv2.line(canvas, (prev_x, prev_y), (ix, iy), draw_color, thickness)
                prev_x, prev_y = ix, iy

            # optional: draw landmarks for debugging (comment out if not needed)
            # mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        else:
            # no hand detected
            prev_x, prev_y = None, None

        # overlay canvas onto frame
        # keep canvas as separate and combine with bitwise to preserve drawn strokes
        gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
        out = cv2.add(frame_bg, canvas_fg)

        # FPS display
        cur_time = time.time()
        fps = 1 / (cur_time - last_time) if cur_time != last_time else 0
        last_time = cur_time
        cv2.putText(out, f"FPS: {int(fps)}", (CAM_WIDTH-140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("AirDraw", out)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s'):  # save snapshot
            save_path = f"airdraw_snapshot_{int(time.time())}.png"
            # merge frame + canvas already in 'out'
            cv2.imwrite(save_path, out)
            print("Saved:", save_path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
