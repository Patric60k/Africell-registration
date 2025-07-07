import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide TensorFlow INFO/WARN

import cv2
import face_recognition
from datetime import datetime
import IdValidation
import LiveFeedValidation
from collections import deque

# Get encoding from the ID image
id_face_encoding = IdValidation.id_image_to_encoding()

# Log setup
log_file = open("Knowledge garden/Soft Eng/Africell registration/logs/face_verification_log.txt", "a", encoding="utf-8")

# Start webcam
cap = cv2.VideoCapture(0)

num_frames = 30  # Liveness buffer size
frames = deque(maxlen=num_frames)
match_counter = 0
required_match_frames = 10

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Collect rolling frames for liveness
    frames.append(frame)

    # Prepare image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)

    # Only run encoding if exactly 1 face is detected (conserves CPU)
    if len(face_locations) != 1:
        name = "Scanning"
        cv2.putText(frame, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    (top, right, bottom, left) = face_locations[0]
    face_encoding = face_encodings[0]
    match = face_recognition.compare_faces([id_face_encoding], face_encoding, tolerance=0.5)[0]

    # Only run liveness check if we have a full frame buffer
    if match and len(frames) == num_frames:
        is_live = LiveFeedValidation.is_live_video(frames, LiveFeedValidation.face_mesh, method="auto")
    else:
        is_live = False

    if match and is_live:
        match_counter += 1
        name = f" Matching... {match_counter}/{required_match_frames}"
        if match_counter >= required_match_frames:
            print("✅✅ FINAL: Face verified & liveness confirmed.")
            name = " MATCH CONFIRMED"
            # Save confirmed face image
            success, snapshot = cap.read()
            if success:
                save_path = f"Knowledge garden/Soft Eng/Africell registration/logs/confirmed_{timestamp.replace(':', '-')}.jpg"
                cv2.imwrite(save_path, snapshot)
                print(f"Face saved to {save_path}")          
            # Log the match  
            log_file.write(f"[{timestamp}] -> {name}\n")
            log_file.flush()
            for _ in range(5):
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.imshow("Face Verification", frame)
                cv2.waitKey(500)
            # Uncomment to auto-exit after match
            break
    else:
        match_counter = 0
        name = "Scanning"

    # Logging
    log_file.write(f"[{timestamp}] -> {name}\n")
    log_file.flush()

    # Draw box
    color = (0, 255, 0) if match else (0, 0, 255)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(frame, f"Match Count: {match_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
log_file.close()
