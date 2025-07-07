import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe face mesh
mpface_mesh = mp.solutions.face_mesh
face_mesh = mpface_mesh.FaceMesh(static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def compute_blink_ratio(landmarks, image_width, image_height):
    # These are the eye landmark indices for right and left eyes (MediaPipe standard)
    LEFT_EYE = [33, 159, 145, 133, 153, 154]  # (outer, top, bottom, inner, lower lid, upper lid)
    RIGHT_EYE = [362, 386, 374, 263, 380, 381]

    def eye_aspect_ratio(indices):
        pts = [landmarks[i] for i in indices]
        p = lambda idx: np.array([pts[idx].x * image_width, pts[idx].y * image_height])

        A = np.linalg.norm(p(1) - p(2))  # top-bottom
        B = np.linalg.norm(p(4) - p(5))  # lid-lid
        C = np.linalg.norm(p(0) - p(3))  # horizontal

        return (A + B) / (2.0 * C) if C != 0 else 0

    left_ratio = eye_aspect_ratio(LEFT_EYE)
    right_ratio = eye_aspect_ratio(RIGHT_EYE)
    return (left_ratio + right_ratio) / 2.0

def is_live_by_motion(frames, face_mesh, motion_threshold=10):
    prev_center = None
    motion_detected = 0

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # Nose tip (landmark 1)
            nose_tip = face_landmarks.landmark[1]
            center = (int(nose_tip.x * frame.shape[1]), int(nose_tip.y * frame.shape[0]))

            if prev_center is not None:
                distance = np.linalg.norm(np.array(center) - np.array(prev_center))
                if distance > motion_threshold:
                    motion_detected += 1

            prev_center = center

    return motion_detected >= 2  # Needs to move twice across N frames

def is_live_by_blink(frames, face_mesh, ear_threshold=0.21, blink_frames_required=2):
    blink_count = 0
    consecutive_low_ear = 0

    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            ear = compute_blink_ratio(face_landmarks.landmark, frame.shape[1], frame.shape[0])

            if ear < ear_threshold:
                consecutive_low_ear += 1
            else:
                if consecutive_low_ear >= 2:
                    blink_count += 1
                consecutive_low_ear = 0

    return blink_count >= blink_frames_required

def is_live_video(frames, face_mesh, method="auto"):
    if method == "motion":
        return is_live_by_motion(frames, face_mesh)
    elif method == "blink":
        return is_live_by_blink(frames, face_mesh)
    elif method == "auto":
        return (
            is_live_by_motion(frames, face_mesh) or
            is_live_by_blink(frames, face_mesh)
        )
    else:
        raise ValueError("Invalid liveness detection method")

