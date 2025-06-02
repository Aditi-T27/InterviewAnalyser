import cv2
import mediapipe as mp
import math

# MediaPipe solutions setup
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Right eye landmarks for blink detection
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def eye_aspect_ratio(eye_points, landmarks):
    # Calculate vertical distances
    A = euclidean_distance(landmarks[eye_points[1]], landmarks[eye_points[5]])
    B = euclidean_distance(landmarks[eye_points[2]], landmarks[eye_points[4]])
    # Calculate horizontal distance
    C = euclidean_distance(landmarks[eye_points[0]], landmarks[eye_points[3]])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection, \
    mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face Detection
        detection_results = face_detection.process(rgb_frame)

        # Face Mesh
        mesh_results = face_mesh.process(rgb_frame)

        # Draw face detection boxes
        if detection_results.detections:
            for detection in detection_results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Process face mesh and blink detection
        if mesh_results.multi_face_landmarks:
            mesh_points = mesh_results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            landmarks = [(int(pt.x * w), int(pt.y * h)) for pt in mesh_points]

            # Draw face mesh landmarks
            mp_drawing.draw_landmarks(
                frame,
                mesh_results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                drawing_spec,
                drawing_spec,
            )

            # Calculate eye aspect ratio (blink detection)
            ear = eye_aspect_ratio(RIGHT_EYE, landmarks)
            if ear < 0.25:
                cv2.putText(frame, "Blinking", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Open Eye", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Combined Face Detection + Mesh + Blink', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
