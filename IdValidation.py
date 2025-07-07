import cv2
import numpy as np
import face_recognition
import pathlib
def id_image_to_encoding():
    base_dir = pathlib.Path(__file__).resolve().parent
    # Ensure the logs directory exists under 'Africell registration/logs'
    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    id_image = face_recognition.load_image_file(images_dir / "id_card.jpg")

    # Convert PIL image to OpenCV BGR format
    id = cv2.cvtColor(id_image, cv2.COLOR_RGB2BGR)
    # Auto-rotate if vertical (for sideways scans) ---
    id = correct_orientation(id, min_face_size=100)
    # Validate ID
    status = validate_id_image(id)
    if status["status"] != "accepted":
        raise ValueError(f"ID validation failed: {status['reason']}")
    
    id_face_encodings = face_recognition.face_encodings(id)
    if len(id_face_encodings) == 0:
        raise ValueError("No face found in ID image.")
    id_face_encoding = id_face_encodings[0]

    id_face_loc = face_recognition.face_locations(id)

    for (top, right, bottom, left), id_face_encoding in zip(id_face_loc, id_face_encodings):
        # Draw bounding box and name
        cv2.rectangle(id, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(id, "Face detected", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,249,0), 2)

    scale_percent = 70  # Reduce original size
    width = int(id.shape[1] * scale_percent / 100)
    height = int(id.shape[0] * scale_percent / 100)
    resized_image = cv2.resize(id, (width, height), interpolation=cv2.INTER_AREA)
#    cv2.imshow("Detected Face", resized_image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()


    return id_face_encoding


def contains_document_shape(image, min_aspect_ratio=1.3, max_aspect_ratio=2.0, area_threshold=0.05):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    image_area = h * w

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w_box, h_box = cv2.boundingRect(approx)
            box_area = w_box * h_box
            aspect_ratio = w_box / h_box if h_box > 0 else 0
            area_ratio = box_area / image_area

            if area_ratio > area_threshold and min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                return True
    return False

def is_blurry(image, threshold=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(variance)
    return variance < threshold

def has_screen_glare(image, brightness_threshold=230, ratio_threshold=1):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bright_pixels = np.sum(v > brightness_threshold)
    total_pixels = v.size
    bright_ratio = bright_pixels / total_pixels
    print(bright_ratio)
    return bright_ratio > ratio_threshold

def validate_id_image(id_image):
    if id_image is None:
        return {"status": "error", "message": "Could not read image"}

    if is_blurry(id_image):
        return {"status": "rejected", "reason": "Image is too blurry"}

    if has_screen_glare(id_image):
        return {"status": "rejected", "reason": "Image likely taken from a screen (glare detected)"}

    h, w = id_image.shape[:2]
    if h < 300 or w < 300:
        return {"status": "rejected", "reason": "Image resolution too low"}

    if not contains_document_shape(id_image):
        return {"status": "rejected", "reason": "No ID document detected in the image"}

    return {"status": "accepted"}

def correct_orientation(id, min_face_size=100):
    best_image = None
    best_face_area = 0

    # Try all 4 rotations
    rotations = [
        ("0", id),
        ("90", cv2.rotate(id, cv2.ROTATE_90_CLOCKWISE)),
        ("180", cv2.rotate(id, cv2.ROTATE_180)),
        ("270", cv2.rotate(id, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

    for label, rotated in rotations:
        face_locations = face_recognition.face_locations(rotated)

        for (top, right, bottom, left) in face_locations:
            width = right - left
            height = bottom - top
            area = width * height

            # Filter out very small faces (false positives like letters)
            if width >= min_face_size and height >= min_face_size:
                print(f"✅ Valid face in {label}° with size: {width}x{height}")
                if area > best_face_area:
                    best_face_area = area
                    best_image = rotated

    if best_image is not None:
        return best_image
    else:
        print("⚠️ No valid face detected in any orientation.")
        return id
