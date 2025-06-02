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
