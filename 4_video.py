import cv2
from transformers import pipeline
from PIL import Image

print("Loading model...")
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

cap = cv2.VideoCapture("test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
print("Processing video... (First 10 frames)")

while cap.isOpened() and frame_count < 10:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    results = detector(pil_image)
    
    print(f"\nFrame {frame_count + 1}:")
    for obj in results:
        print(f"  - Object: {obj['label']}, Score: {obj['score']:.2f}")
    
    frame_count += 1

cap.release()
print("\nProcessing complete.")