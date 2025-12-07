import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------
# LOAD MODELS
# ---------------------------
face_detector = YOLO("yolov8n-face.onnx")

classifier = ort.InferenceSession(
    "best_model-v3.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Preprocessing for EfficientNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# smoothing
alpha = 0.25
smooth_prob = 0.0


def classify_face(face):
    img = Image.fromarray(face)
    x = preprocess(img).unsqueeze(0).numpy()
    logits = classifier.run(None, {"input": x})[0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    return float(probs[0][1])  # fake probability


# ---------------------------
# VIDEO PROCESSING LOOP
# ---------------------------
def predict_video(path):
    global smooth_prob

    cap = cv2.VideoCapture(path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        detections = face_detector(frame)[0]

        fake_values = []

        for det in detections.boxes:
            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            fake_prob = classify_face(face)
            fake_values.append(fake_prob)

        # Temporal smoothing
        if fake_values:
            avg_fake = np.mean(fake_values)
            smooth_prob = (1 - alpha) * smooth_prob + alpha * avg_fake

        # ---------------------------
        # DRAW RED / GREEN INDICATOR
        # ---------------------------
        color = (0, 255, 0) if smooth_prob < 0.5 else (0, 0, 255)
        label = f"Fake Prob: {smooth_prob:.2f}"

        # Dot
        cv2.circle(frame, (40, 40), 20, color, -1)

        # Text
        cv2.putText(frame, label, (70, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        # Show window
        cv2.imshow("Jetson Deepfake Detector", frame)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------
# RUN
# ---------------------------
predict_video("test_video.mp4")
