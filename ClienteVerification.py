import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hides INFO and WARNING logs from TensorFlow

import cv2
import face_recognition
from datetime import datetime
import IdValidation
import LiveFeedValidation
from collections import deque
from playsound import playsound
from pathlib import Path

# Get face encoding from ID image
id_face_encoding = IdValidation.id_image_to_encoding()

# Get base directory of the current script
base_dir = Path(__file__).resolve().parent

# Ensure the logs directory exists under 'Africell registration/logs'
logs_dir = base_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
# Open a log file for writing verification results
log_file = open(logs_dir / "face_verification_log.txt", "a", encoding="utf-8")

# Ensure the sounds directory exists under 'Africell registration/sounds'
sounds_dir = base_dir / "sounds"
sounds_dir.mkdir(parents=True, exist_ok=True)

# Start webcam capture
cap = cv2.VideoCapture(0)

num_frames = 30  # Number of frames to collect for liveness
frames = deque(maxlen=num_frames)  # Automatically discards old frames
match_counter = 0  # Counts consecutive matches
required_match_frames = 10  # Number of matches required for confirmation

while True:
    ret, frame = cap.read()
    if not ret:
        continue  # Skip if frame not captured

    frames.append(frame)
    # Convert frame to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detect face locations and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Loop through detected faces
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected face with ID face encoding
        matches = face_recognition.compare_faces([id_face_encoding], face_encoding, tolerance=0.5)
        match = True in matches

        # Check for match and liveness
        if match and LiveFeedValidation.is_live_video(frames, LiveFeedValidation.face_mesh, method="auto"):
            match_counter += 1
            name = f" Matching... {match_counter}/{required_match_frames}"
            if match_counter >= required_match_frames:                
                print("✅✅ FINAL: Face verified & liveness confirmed.")
                name = f" MATCH CONFIRMED"
                # Log the match
                log_file.write(f"[{timestamp}] -> MATCH CONFIRMED\n")
                log_file.flush()
                # Save confirmed face image
                success, snapshot = cap.read()
                if success:
                    save_path = logs_dir / f"confirmed_{timestamp.replace(':', '-')}.jpg"
                    cv2.imwrite(str(save_path), snapshot)
                # Play confirmation sound                
                try:
                    playsound(sounds_dir / "match_confirmed.wav")
                except Exception as e:
                    print("Couldn't play audio:", e)

                for _ in range(5):
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                    cv2.imshow("Face Verification", frame)
                    cv2.waitKey(500)
                # Uncomment below to stop after match is confirmed
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
        else:
            match_counter = 0  # Reset counter if not matching or not live
            name = "Scanning"

        # Log the result for this frame
        log_file.write(f"[{timestamp}] -> {name}\n")
        log_file.flush()

        # Draw bounding box and label on the frame
        color = (0, 255, 0) if match else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)


    # Display the frame with annotations
    cv2.imshow("Face Verification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop if 'q' is pressed

# Release resources and close files/windows
cap.release()
cv2.destroyAllWindows()
log_file.close()
