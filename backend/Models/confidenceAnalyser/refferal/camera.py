import cv2

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Display the resulting frame
    cv2.imshow("Live Video - Press 'q' to Quit", frame)

    # Break loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close window
cap.release()
cv2.destroyAllWindows()
