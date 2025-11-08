## 🧠 Project Name: **Air Draw using Hand Gestures**

### 🎯 Project Concept:

Ye project ek **Computer Vision-based application** hai jisme user **apne haath se hawa me drawing** kar sakta hai — webcam ke through!
Yani mouse ya touchscreen ke bina, sirf finger movement se drawing banayi ja sakti hai.

---

## ⚙️ Step 1: Libraries Used — Explanation

| Library                     | Use / Explanation                                                                                                                         |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenCV (`cv2`)**          | Image/Video processing ke liye use hoti hai. Isse webcam se frame capture karte hain aur uspar drawing karte hain.                        |
| **NumPy (`numpy`)**         | Pixel arrays handle karne ke liye. Frame ko matrix ke form me treat karta hai (image manipulations ke liye).                              |
| **Mediapipe (`mediapipe`)** | Google ka AI library hai jo hand detection aur landmark tracking provide karti hai. Ye har finger ke joints (21 points) detect karta hai. |

---

## 💻 Step 2: Project Structure

```
air-draw/
│
├── air_draw.py              # main application file
├── .venv/               # virtual environment folder
├── requirements.txt     # dependencies list (optional)
└── README.md            # (optional) project info
```

---

## 🧩 Step 3: Code Explanation — Line by Line

### 🧱 Import Libraries

```python
import cv2
import mediapipe as mp
import numpy as np
```

* Ye 3 core libraries import kar rahe hain:

  * `cv2` → webcam aur drawing ke liye
  * `mp` → hand detection ke liye
  * `np` → image arrays handle karne ke liye

---

### ✋ Initialize Mediapipe Hand Detector

```python
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
```

* `mp_hands.Hands()` ek **AI model** initialize karta hai jo live video se hand detect karta hai.
* `mp_drawing` optional hai — agar aapko hand skeleton draw karna ho to.

---

### 🎥 Capture Video from Webcam

```python
cap = cv2.VideoCapture(0)
```

* `0` means default webcam.
* Ye continuously webcam se frames capture karega.

---

### 🧾 Create a Blank Canvas

```python
canvas = None
```

* Ye ek blank image (same size as webcam frame) store karega jisme hum apni drawing save karenge.

---

### 🖐 Main Loop — Process Each Frame

```python
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
```

* `ret, frame = cap.read()` → ek frame capture karta hai.
* `cv2.flip(frame, 1)` → mirror effect deta hai (left/right swap).

---

### 🤖 Convert Frame to RGB (for Mediapipe)

```python
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
result = hands.process(rgb)
```

* Mediapipe ko **RGB format** me image chahiye hoti hai.
* `result` me hand landmarks detect ho kar milte hain (agar haath dikhe to).

---

### ✏️ Initialize Canvas Size

```python
if canvas is None:
    canvas = np.zeros_like(frame)
```

* Pehle frame ke size ke equal ek black image create karta hai.
* Isme hum apni drawing karte hain.

---

### 🧍 Detect Hand Landmarks

```python
if result.multi_hand_landmarks:
    for hand_landmarks in result.multi_hand_landmarks:
        index_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
        index_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
```

* `landmark[8]` → index finger ka **tip point** hota hai.
* Uske `x` aur `y` coordinate ko frame ke size ke hisaab se scale kar rahe hain.

---

### 🎨 Draw Circle at Finger Tip

```python
cv2.circle(frame, (index_x, index_y), 10, (0, 0, 255), -1)
```

* Ye finger tip par red circle draw karta hai (pointer jaisa).

---

### 🖋️ Draw Line as User Moves Finger

```python
if prev_x is not None and prev_y is not None:
    cv2.line(canvas, (prev_x, prev_y), (index_x, index_y), (255, 0, 0), 5)
```

* Jab finger move hoti hai, to previous aur current position ke beech ek **blue line** draw hoti hai.
* Isse lagta hai jaise aap hawa me drawing kar rahe ho.

---

### 🔄 Combine Canvas with Frame

```python
frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
```

* Ye dono ko mix karta hai taaki drawing visible ho frame ke upar.

---

### 👀 Display the Frame

```python
cv2.imshow("Air Draw", frame)
```

* Ye ek window open karta hai jisme drawing live visible hoti hai.

---

### 🛑 Exit on 'q' Press

```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

* Jab user ‘q’ dabata hai to loop break ho jata hai aur program stop ho jata hai.

---

### 🔒 Release Resources

```python
cap.release()
cv2.destroyAllWindows()
```

* Webcam aur windows properly close karta hai (memory clean-up).

---

## 🖼️ Final Output

✅ User webcam ke samne apna haath rakhta hai.
✅ System index finger detect karta hai.
✅ Finger move karte hi line draw hoti hai.
✅ `q` dabane par window close ho jati hai.

---

## 💡 Bonus Tips (Explaination ke time)

| Question                              | Short Answer                                                                                    |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- |
| **Why use Mediapipe?**                | Because it provides fast, real-time hand tracking using deep learning models optimized for CPU. |
| **What’s the FPS?**                   | Around 30–60 FPS on normal systems.                                                             |
| **Can we add color change / eraser?** | Yes, by adding button zones or gesture conditions (like 2-finger for eraser).                   |
| **Can this run on Raspberry Pi?**     | Yes, but slower — needs optimization.                                                           |

---
1st 
"""Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass"""
2nd
""".\.venv\Scripts\Activate.ps1"""
3rd
"""python air_draw.py"""