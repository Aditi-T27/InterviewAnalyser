import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh and Pose solutions once
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_landmark_coordinates(landmarks, image_shape, indices):
    """
    Extract pixel coordinates for the given landmark indices from normalized landmarks.
    
    Args:
        landmarks: mediapipe landmarks object
        image_shape: (height, width) tuple of image
        indices: list of int indices for landmarks
    
    Returns:
        coords: np.array shape (len(indices), 2) with (x, y) pixel coords
    """
    h, w = image_shape[:2]
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    return np.array(coords)

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) from 6 eye landmarks.
    
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    """
    p = landmarks[eye_indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(landmarks, mouth_indices):
    """
    Calculate Mouth Aspect Ratio (MAR).
    
    MAR = (||p3 - p9|| + ||p4 - p8|| + ||p5 - p7||) / (2 * ||p0 - p6||)
    """
    p = landmarks[mouth_indices]
    A = np.linalg.norm(p[2] - p[8])
    B = np.linalg.norm(p[3] - p[7])
    C = np.linalg.norm(p[4] - p[6])
    D = np.linalg.norm(p[0] - p[6])
    mar = (A + B + C) / (2.0 * D)
    return mar

def eyebrow_raise(landmarks, eye_center_idx, brow_center_idx, image_shape):
    """
    Calculate normalized vertical distance between eyebrow and eye center as a proxy for eyebrow raise.
    
    Returns a ratio normalized by image height.
    """
    h, w = image_shape[:2]
    eye_y = landmarks[eye_center_idx].y * h
    brow_y = landmarks[brow_center_idx].y * h
    return (eye_y - brow_y) / h  # positive means brow above eye

def get_head_pose(pose_landmarks, image_shape):
    """
    Calculate approximate head pose (pitch, yaw, roll) from MediaPipe Pose landmarks.
    
    Uses nose, shoulders, and eyes for estimation.
    Returns angles in degrees.
    """
    if pose_landmarks is None:
        return 0.0, 0.0, 0.0
    
    h, w = image_shape[:2]

    # Helper to get 2D pixel points for landmarks
    def lm_to_point(idx):
        lm = pose_landmarks[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)
    
    left_shoulder = lm_to_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    right_shoulder = lm_to_point(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    nose = lm_to_point(mp_pose.PoseLandmark.NOSE.value)
    left_eye = lm_to_point(mp_pose.PoseLandmark.LEFT_EYE.value)
    right_eye = lm_to_point(mp_pose.PoseLandmark.RIGHT_EYE.value)

    # Calculate vectors
    shoulder_vec = right_shoulder - left_shoulder
    shoulder_angle = math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0]))
    roll = shoulder_angle - 180  # rotation of head along Z-axis

    nose_to_left_eye = left_eye - nose
    nose_to_right_eye = right_eye - nose
    eye_line_angle = math.degrees(math.atan2(nose_to_left_eye[1], nose_to_left_eye[0]))
    yaw = eye_line_angle - 90  # yaw estimation

    pitch = (nose[1] - ((left_shoulder[1] + right_shoulder[1]) / 2)) / h * 90  # approximate pitch
    
    return pitch, yaw, roll

def process_face_mesh(frame):
    """
    Run MediaPipe FaceMesh on the frame and return landmarks.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks[0].landmark
    return None

def process_pose(frame):
    """
    Run MediaPipe Pose on the frame and return landmarks.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    return None
def calculate_gaze_score(iris_coords, eye_corner_left, eye_corner_right):
    """
    Estimate gaze direction based on iris position relative to eye corners.
    Return score between 0 (looking away) and 1 (looking straight).
    """
    eye_width = np.linalg.norm(eye_corner_right - eye_corner_left)
    iris_dist = np.linalg.norm(iris_coords - ((eye_corner_left + eye_corner_right) / 2))

    # Normalize
    gaze_score = 1 - (iris_dist / (eye_width / 2))
    return max(0.0, min(1.0, gaze_score))


# def calculate_confidence(ear, mar, eyebrow, pitch, yaw, roll):
#     """
#     Calculate a confidence score based on facial metrics.
#     Score is between 0 and 100.
#     """
#     score = 0

#     # EAR: Good if between 0.2 and 0.3
#     if 0.2 < ear < 0.3:
#         score += 25

#     # MAR: Good if between 0.3 and 0.6 (speaking mildly or neutral mouth)
#     if 0.3 < mar < 0.6:
#         score += 25

#     # Eyebrow Raise: ~0.02 to 0.06 (not too tense, not too relaxed)
#     if 0.02 < eyebrow < 0.06:
#         score += 25

#     # Head Pose: Close to neutral
#     if abs(pitch) < 10 and abs(yaw) < 10 and abs(roll) < 10:
#         score += 25

#     return score

# def calculate_confidence(avg_ear, mar, eyebrow_raise_val, pitch, yaw, roll):
#     confidence = 100  # Start at full confidence

#     # EAR: Eye Aspect Ratio
#     if avg_ear < 0.21:
#         confidence -= 20  # Possibly blinking or eyes mostly closed

#     # Head orientation
#     if abs(pitch) > 45 or abs(yaw) > 45:
#         confidence -= 25  # Not facing camera directly

#     # MAR: Mouth Aspect Ratio
#     if mar < 0.4:
#         confidence -= 15  # Closed mouth — possibly not smiling/speaking

#     # Eyebrow raise: if too low, reduce confidence (no expressiveness)
#     if eyebrow_raise_val < 0.03:
#         confidence -= 10

#     # Clamp confidence between 0 and 100
#     confidence = max(0, min(100, confidence))

#     return int(confidence)
def calculate_confidence(avg_ear, mar, eyebrow_raise_val, pitch, yaw, roll, gaze_score):
    """
    Calculate confidence score based on facial metrics and gaze direction.
    
    Parameters:
    - avg_ear: Average Eye Aspect Ratio
    - mar: Mouth Aspect Ratio
    - eyebrow_raise_val: Eyebrow raise metric
    - pitch, yaw, roll: Head orientation angles in degrees
    - gaze_score: 1.0 if looking straight, <1 if looking away (e.g. 0.0–1.0)

    Returns:
    - confidence: int (0 to 100)
    """

    confidence = 100  # Start at max

    # ------------------- Eye Blink (EAR) -------------------
    if avg_ear < 0.21:
        confidence -= 20  # Eyes closed or blinking too much

    # ------------------- Head Pose -------------------
    if abs(pitch) > 30 or abs(yaw) > 30 or abs(roll) > 25:
        confidence -= 20  # Not facing the screen properly

    # ------------------- Mouth (Smile Detection) -------------------
    if mar > 0.6:
        confidence += 10  # Smiling boosts confidence
    elif mar < 0.3:
        confidence -= 10  # Neutral or tight lips reduces confidence

    # ------------------- Eyebrow Movement -------------------
    if eyebrow_raise_val < 0.02:
        confidence -= 10  # No expressiveness

    # ------------------- Gaze -------------------
    if gaze_score < 0.8:
        confidence -= int((1 - gaze_score) * 20)  # Penalize looking away

    # Clamp
    confidence = max(0, min(100, confidence))

    return int(confidence)

