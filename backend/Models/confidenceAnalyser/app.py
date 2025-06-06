# # import cv2
# # from utils.mediapipe_helpers import (
# #     process_face_mesh,
# #     process_pose,
# #     get_landmark_coordinates,
# #     eye_aspect_ratio,
# #     mouth_aspect_ratio,
# #     eyebrow_raise,
# #     get_head_pose,
# #     calculate_confidence,
# #     calculate_gaze_score
# # )
# # import numpy as np

# # # Landmark indices for left eye (MediaPipe Face Mesh)
# # LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # typical 6 eye landmarks for EAR
# # # Landmark indices for right eye
# # RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# # # Mouth indices for MAR (example 8 points around mouth)
# # MOUTH_INDICES = [61, 81, 311, 291, 308, 324, 78, 95, 88, 178]  # simplified example

# # # For eyebrow raise, example indices (brow center and eye center)
# # LEFT_EYE_CENTER_IDX = 159  # approximate center upper eye
# # LEFT_BROW_CENTER_IDX = 105  # approximate brow center

# # def main():
# #     cap = cv2.VideoCapture(0)
# #     if not cap.isOpened():
# #         print("Cannot open webcam")
# #         return

# #     while True:
# #         ret, frame = cap.read()
# #         if not ret:
# #             break

# #         image_shape = frame.shape

# #         # Process face mesh
# #         face_landmarks = process_face_mesh(frame)

# #         if face_landmarks:
# #             # Get pixel coords for left and right eye landmarks
# #             left_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, LEFT_EYE_INDICES)
# #             right_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, RIGHT_EYE_INDICES)

# #             # Calculate EAR for both eyes
# #             left_ear = eye_aspect_ratio(left_eye_pts, list(range(6)))
# #             right_ear = eye_aspect_ratio(right_eye_pts, list(range(6)))
# #             avg_ear = (left_ear + right_ear) / 2

# #             # Get mouth points and calculate MAR
# #             mouth_pts = get_landmark_coordinates(face_landmarks, image_shape, MOUTH_INDICES[:10])
# #             mar = mouth_aspect_ratio(mouth_pts, list(range(10)))

# #             # Eyebrow raise (normalized vertical distance)
# #             eyebrow_raise_val = eyebrow_raise(face_landmarks, LEFT_EYE_CENTER_IDX, LEFT_BROW_CENTER_IDX, image_shape)

# #             # Draw eye landmarks
# #             for (x, y) in left_eye_pts:
# #                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
# #             for (x, y) in right_eye_pts:
# #                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

# #             # Draw mouth landmarks
# #             for (x, y) in mouth_pts:
# #                 cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

# #             # Display computed metrics on frame
# #             cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #             cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
# #             cv2.putText(frame, f"Eyebrow Raise: {eyebrow_raise_val:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)


# #         # Process pose for head pose
# #         pose_landmarks = process_pose(frame)
# #         pitch, yaw, roll = get_head_pose(pose_landmarks, image_shape)
# #         cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #         cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #         cv2.putText(frame, f"Roll: {roll:.1f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
# #         confidence_score = calculate_confidence(avg_ear, mar, eyebrow_raise_val, pitch, yaw, roll)
# #         # print(mar)
# #         cv2.putText(frame, f"Confidence: {confidence_score}%", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# #         # Show the frame
# #         cv2.imshow("Confidence Analysis - Press Q to Quit", frame)

# #         if cv2.waitKey(5) & 0xFF == ord('q'):
# #             break

# #     cap.release()
# #     cv2.destroyAllWindows()

# # if __name__ == "__main__":
# #     main()


# import cv2
# from utils.mediapipe_helpers import (
#     process_face_mesh,
#     process_pose,
#     get_landmark_coordinates,
#     eye_aspect_ratio,
#     mouth_aspect_ratio,
#     eyebrow_raise,
#     get_head_pose,
#     calculate_confidence,
#     calculate_gaze_score
# )
# import numpy as np

# # Landmark indices for left eye (MediaPipe Face Mesh)
# LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # typical 6 eye landmarks for EAR
# # Landmark indices for right eye
# RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# # Mouth indices for MAR (example 8 points around mouth)
# MOUTH_INDICES = [61, 81, 311, 291, 308, 324, 78, 95, 88, 178]  # simplified example

# # For eyebrow raise, example indices (brow center and eye center)
# LEFT_EYE_CENTER_IDX = 159  # approximate center upper eye
# LEFT_BROW_CENTER_IDX = 105  # approximate brow center

# # Gaze indices
# LEFT_IRIS_IDX = 468
# RIGHT_IRIS_IDX = 473
# LEFT_EYE_CORNER = [33, 133]    # Inner and outer corners
# RIGHT_EYE_CORNER = [362, 263]

# def main():
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Cannot open webcam")
#         return

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         image_shape = frame.shape

#         # Process face mesh
#         face_landmarks = process_face_mesh(frame)

#         if face_landmarks:
#             # Get pixel coords for left and right eye landmarks
#             left_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, LEFT_EYE_INDICES)
#             right_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, RIGHT_EYE_INDICES)

#             # Calculate EAR for both eyes
#             left_ear = eye_aspect_ratio(left_eye_pts, list(range(6)))
#             right_ear = eye_aspect_ratio(right_eye_pts, list(range(6)))
#             avg_ear = (left_ear + right_ear) / 2

#             # Get mouth points and calculate MAR
#             mouth_pts = get_landmark_coordinates(face_landmarks, image_shape, MOUTH_INDICES[:10])
#             mar = mouth_aspect_ratio(mouth_pts, list(range(10)))

#             # Eyebrow raise (normalized vertical distance)
#             eyebrow_raise_val = eyebrow_raise(face_landmarks, LEFT_EYE_CENTER_IDX, LEFT_BROW_CENTER_IDX, image_shape)

#             # === Gaze Tracking ===
#             gaze_indices = [LEFT_IRIS_IDX, RIGHT_IRIS_IDX] + LEFT_EYE_CORNER + RIGHT_EYE_CORNER
#             gaze_coords = get_landmark_coordinates(face_landmarks, image_shape, gaze_indices)

#             gaze_idx_map = {idx: i for i, idx in enumerate(gaze_indices)}

#             left_gaze_score = calculate_gaze_score(
#                 gaze_coords[gaze_idx_map[LEFT_IRIS_IDX]],
#                 gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[0]]],
#                 gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[1]]]
#             )

#             right_gaze_score = calculate_gaze_score(
#                 gaze_coords[gaze_idx_map[RIGHT_IRIS_IDX]],
#                 gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[0]]],
#                 gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[1]]]
#             )

#             gaze_score = (left_gaze_score + right_gaze_score) / 2

#             # Draw eye landmarks
#             for (x, y) in left_eye_pts:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#             for (x, y) in right_eye_pts:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

#             # Draw mouth landmarks
#             for (x, y) in mouth_pts:
#                 cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

#             # Display computed metrics on frame
#             cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#             cv2.putText(frame, f"Eyebrow Raise: {eyebrow_raise_val:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
#             cv2.putText(frame, f"Gaze Score: {gaze_score:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)


#         # Process pose for head pose
#         pose_landmarks = process_pose(frame)
#         pitch, yaw, roll = get_head_pose(pose_landmarks, image_shape)
#         cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#         cv2.putText(frame, f"Roll: {roll:.1f}", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

#         # Calculate confidence score
#         confidence_score = calculate_confidence(avg_ear, mar, eyebrow_raise_val, pitch, yaw, roll, gaze_score)
#         cv2.putText(frame, f"Confidence: {confidence_score}%", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

#         # Show the frame
#         cv2.imshow("Confidence Analysis - Press Q to Quit", frame)

#         if cv2.waitKey(5) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


import cv2
import time
from utils.mediapipe_helpers import (
    process_face_mesh,
    process_pose,
    get_landmark_coordinates,
    eye_aspect_ratio,
    mouth_aspect_ratio,
    eyebrow_raise,
    get_head_pose,
    calculate_confidence,
    calculate_gaze_score
)
import numpy as np

# EAR, MAR, Eyebrow, Iris landmarks
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [61, 81, 311, 291, 308, 324, 78, 95, 88, 178]
LEFT_EYE_CENTER_IDX = 159
LEFT_BROW_CENTER_IDX = 105
LEFT_IRIS_IDX = 468
RIGHT_IRIS_IDX = 473
LEFT_EYE_CORNER = [33, 133]
RIGHT_EYE_CORNER = [362, 263]

# Blink detection thresholds
BLINK_EAR_THRESHOLD = 0.21
BLINK_COOLDOWN_SECONDS = 0.1

def main():
    cap = cv2.VideoCapture(0)
    confidence_scores = []

    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    blink_count = 0
    last_blink_time = time.time()
    blink_start = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_shape = frame.shape
        pose_landmarks = process_pose(frame)
        pitch, yaw, roll = get_head_pose(pose_landmarks, image_shape)

        face_landmarks = process_face_mesh(frame)
        distracted = False
        blink_now = False

        if face_landmarks:
            # --- EAR (Eye Aspect Ratio) ---
            left_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, LEFT_EYE_INDICES)
            right_eye_pts = get_landmark_coordinates(face_landmarks, image_shape, RIGHT_EYE_INDICES)
            left_ear = eye_aspect_ratio(left_eye_pts, list(range(6)))
            right_ear = eye_aspect_ratio(right_eye_pts, list(range(6)))
            avg_ear = (left_ear + right_ear) / 2
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # --- Blink Detection ---
            if avg_ear < BLINK_EAR_THRESHOLD:
                now = time.time()
                if now - last_blink_time > BLINK_COOLDOWN_SECONDS:
                    blink_count += 1
                    last_blink_time = now
                    blink_now = True

            if blink_count >= 20:
                cv2.putText(frame, "You seem nervous. Try to calm down.", (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if blink_count <= 10:
                cv2.putText(frame, "Good to Go", (30, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # --- MAR ---
            mouth_pts = get_landmark_coordinates(face_landmarks, image_shape, MOUTH_INDICES)
            mar = mouth_aspect_ratio(mouth_pts, list(range(10)))
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            # --- Eyebrow Raise ---
            eyebrow_val = eyebrow_raise(face_landmarks, LEFT_EYE_CENTER_IDX, LEFT_BROW_CENTER_IDX, image_shape)
            cv2.putText(frame, f"Eyebrow: {eyebrow_val:.3f}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,255), 2)

            # --- Gaze ---
            gaze_indices = [LEFT_IRIS_IDX, RIGHT_IRIS_IDX] + LEFT_EYE_CORNER + RIGHT_EYE_CORNER
            gaze_coords = get_landmark_coordinates(face_landmarks, image_shape, gaze_indices)
            gaze_idx_map = {idx: i for i, idx in enumerate(gaze_indices)}

            left_gaze_score = calculate_gaze_score(
                gaze_coords[gaze_idx_map[LEFT_IRIS_IDX]],
                gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[0]]],
                gaze_coords[gaze_idx_map[LEFT_EYE_CORNER[1]]]
            )
            right_gaze_score = calculate_gaze_score(
                gaze_coords[gaze_idx_map[RIGHT_IRIS_IDX]],
                gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[0]]],
                gaze_coords[gaze_idx_map[RIGHT_EYE_CORNER[1]]]
            )
            gaze_score = (left_gaze_score + right_gaze_score) / 2.0
            cv2.putText(frame, f"Gaze: {gaze_score:.2f}", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

            # --- Distraction detection ---
            if  gaze_score < 0.5:
                distracted = True
                cv2.putText(frame, "You seem distracted. Please focus.", (30, 430),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

            # --- Final Confidence Score ---
            confidence_score = calculate_confidence(avg_ear, mar, eyebrow_val, pitch, yaw, roll, gaze_score)
            cv2.putText(frame, f"Confidence: {confidence_score}%", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            confidence_scores.append(confidence_score)


        # --- Head pose display ---
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Yaw: {yaw:.1f}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.putText(frame, f"Roll: {roll:.1f}", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # Show final output
        cv2.imshow("🧠 Confidence & Focus Analyzer - Press Q to Quit", frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if confidence_scores:
     average_confidence = sum(confidence_scores) / len(confidence_scores)
     print(f"\nAverage Confidence Score for the Session: {average_confidence:.2f}%")


if __name__ == "__main__":
    main()
