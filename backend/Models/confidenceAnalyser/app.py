import cv2
from utils.mediapipe_helpers import (
    process_face_mesh,
    process_pose,
    get_landmark_coordinates,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    eyebrow_raise,
    get_head_pose,
)
import numpy as np

# Landmark indices for left eye (MediaPipe Face Mesh)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # typical 6 eye landmarks for EAR
# Landmark indices for right eye
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mouth indices for MAR (example 8 points around mouth)
MOUTH_INDICES = [61, 81, 311, 291, 308, 324, 78, 95, 88, 178]  # simplified example

# For eyebrow raise, example indices (brow center and eye center)
LEFT_EYE_CENTER_IDX = 159  # approximate center upper eye
LEFT_BROW_CENTER_IDX = 105  # approximate brow center

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_shape = frame.shape

        # Process face mesh
        face_landmarks = process_face_mesh(frame)

        if face_landmarks:
            # Get pixel coords for left and right eye landmarks
            left_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, LEFT_EYE_INDICES)
            right_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, RIGHT_EYE_INDICES)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye_pts, list(range(6)))
            right_ear = eye_aspect_ratio(right_eye_pts, list(range(6)))
            avg_ear = (left_ear + right_ear) / 2

            # Get mouth points and calculate MAR
            mouth_pts = get_landmark_coordinates(face_landmarks, image_shape, MOUTH_INDICES[:10])
            mar = mouth_aspect_ratio(mouth_pts, list(range(10)))

            # Eyebrow raise (normalized vertical distance)
            eyebrow_raise_val = eyebrow_raise(face_landmarks, LEFT_EYE_CENTER_IDX, LEFT_BROW_CENTER_IDX, image_shape)

            # Draw eye landmarks
            for (x, y) in left_eye_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            for (x, y) in right_eye_pts:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Draw mouth landmarks
            for (x, y) in mouth_pts:
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Display computed metrics on frame
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Eyebrow Raise: {eyebrow_raise_val:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

        # Process pose for head pose
        pose_landmarks = process_pose(frame)
        pitch, yaw, roll = get_head_pose(pose_landmarks, image_shape)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Show the frame
        cv2.imshow("Confidence Analysis - Press Q to Quit", frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
