import cv2
import numpy as np
import cvzone

# Fetch the video
cap = cv2.VideoCapture(r"C:\Users\dubey\Downloads\1900-151662242_small.mp4")

# Counting Line position
count_line_pos = 550

# Minimum width & height of rectangle over each vehicle
min_width = 80
min_height = 80

# Initialize the Background Subtractor Algorithm (MOG2)
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

# To draw the circle passing which the vehicle would be counted
def centre_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# Initialize variables for counting
offset = 3
counter = 0
vehicle_crossed = set()  # Set to track vehicles already counted
vehicle_positions = []   # List to track the center of each vehicle

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    # Convert to grayscale and apply Gaussian blur
    imgGray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 5)

    # Apply background subtraction
    img_sub = algo.apply(imgBlur)
    imgDilate = cv2.dilate(img_sub, np.ones((3, 3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    dilat = cv2.morphologyEx(imgDilate, cv2.MORPH_CLOSE, kernel)
    dilat = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

    contour_shape, h = cv2.findContours(dilat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Drawing the counter line
    cv2.line(frame1, (25, count_line_pos), (1200, count_line_pos), (0, 255, 255), 5)

    detections = []  # List to store detected vehicles

    for i, c in enumerate(contour_shape):
        (x, y, w, h) = cv2.boundingRect(c)

        # Only consider contours that are large enough to be vehicles
        if w >= min_width and h >= min_height:
            # Store the bounding box for vehicle detection
            detections.append([x, y, x + w, y + h])

            # Draw the rectangle around the detected vehicle
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # Calculate the center of the vehicle
            centre = centre_handle(x, y, w, h)
            vehicle_positions.append(centre)
            cv2.circle(frame1, centre, 4, (0, 0, 255), -1)

    # Process detections and check for crossing the counting line
    for centre in vehicle_positions:
        cx, cy = centre

        # Only count vehicles that cross the line once (avoid multiple counts)
        if (count_line_pos + offset) > cy > (count_line_pos - offset):
            if centre not in vehicle_crossed:
                counter += 1
                vehicle_crossed.add(centre)

    # Reduce the size of the vehicle count text and position it in the left half
    font_scale = 1  # Reduced font scale
    thickness = 2
    text = f"Vehicle Count: {counter}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, thickness)[0]

    # Position text on the left half of the frame
    text_x = 25
    text_y = 70
    cv2.putText(frame1, text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (255, 0, 0), thickness)

    # Show the frame
    cv2.imshow("Vehicle Counting", frame1)

    # Break on pressing Enter
    if cv2.waitKey(10) == 13:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
