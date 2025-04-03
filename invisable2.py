import cv2
import numpy as np
import time
import os

# Create a folder to save the output if not exists
output_folder = "Invisible_Cloak_Output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start video capture
cap = cv2.VideoCapture(0)

# Get video frame width and height
frame_width = int(cap.get(10))
frame_height = int(cap.get(11))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = os.path.join(output_folder, "invisible_cloak_output.avi")
out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

# Allow camera to warm up
time.sleep(2)

# Step 1: Capture Cloth Color Dynamically (5 seconds)
print("Hold the cloth steady in the center for color detection...")
start_time = time.time()
detected_color = None

while time.time() - start_time < 5:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w, _ = frame.shape
    center_x, center_y = w // 2, h // 2
    region = hsv[center_y - 10:center_y + 10, center_x - 10:center_x + 10]

    avg_color = np.mean(region, axis=(0, 1)).astype(int)

    lower_color = np.array([max(avg_color[0] - 10, 0), 40, 40])
    upper_color = np.array([min(avg_color[0] + 10, 179), 255, 255])

    detected_color = (lower_color, upper_color)

if detected_color:
    print(f"Color combination detected: Lower={detected_color[0]}, Upper={detected_color[1]}")
    print("Color combination taken successfully!")
else:
    print("Color combination not taken!")

# Step 2: Capture Background Image (6 seconds)
print("Stand away from the frame for background detection...")
time.sleep(1)
background = None
start_time = time.time()

while time.time() - start_time < 6:
    ret, background = cap.read()

background = np.flip(background, axis=1)
print("Background image detected!")
print("Recording started...")

# Step 3: Apply Cloak Effect and Save Video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_color, upper_color = detected_color
    mask = cv2.inRange(hsv, lower_color, upper_color)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    mask_inv = cv2.bitwise_not(mask)

    res1 = cv2.bitwise_and(background, background, mask=mask)
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Write frame to video file
    out.write(final_output)

    cv2.imshow("Invisible Cloak", final_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Recording saved successfully at: {output_path}")

