# SWE Prime 413 - Machine Learning Applications

## Project Overview 
This project shows how to use pre-trained machine learning models to solve four different tasks:

1. Text Processing (Sentiment Analysis)
2. Audio Processing (Speech-to-Text)
3. Image Processing (Image Classification)
4. Video Processing (Object Detection)

**Important:** No model training was done. All models are pre-trained and used directly. This is called "inference".

**Note:** The LLM (Large Language Model) task could not be completed because of internet download problems. Details are at the end of this report.

## Summary Table

| Task | Library | Model | Status |
|------|---------|-------|--------|
| Text Processing | Hugging Face | DistilBERT | Complete |
| Audio Processing | Hugging Face | Wav2Vec2 | Complete |
| Image Processing | PyTorch | ResNet50 | Complete |
| Video Processing | Hugging Face + OpenCV | DETR-ResNet50 | Complete |
| LLM | llama-cpp-python | Llama-2-7B | Not Completed (Download Issue) |

## Frameworks Used
- Hugging Face Transformers: Text, Audio, Video
- PyTorch: Image
- OpenCV: Video frame extraction

## Conclusion
This project successfully used pre-trained models for four different tasks. The only problem was the local LLM, which could not be downloaded because of internet issues.

## How to Run
1. Install Python 3.12.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install libraries: `pip install transformers torch torchvision opencv-python soundfile pillow`
5. Run each script:
   - `python 1_text.py`
   - `python 2_audio.py`
   - `python 3_image.py`
   - `python 4_video.py`


### Task 1, (1_text.py): Text Processing (Sentiment Analysis)

### Library Used 
- Hugging Face Transformers (`transformers`)

### Model Used 
- `distilbert-base-uncased-finetuned-sst-2-english`
- DistilBERT is a small and fast version of BERT.
- It is trained to classify text as POSITIVE or NEGATIVE.

### How the Model Works 
1. The input text is broken into small pieces called tokens.
2. Each token is converted into a number (token ID).
3. The numbers go through a neural network with attention layers.
4. The model outputs two probabilities: POSITIVE and NEGATIVE.
5. The higher probability is chosen as the result.

### CODE 
```python
from transformers import pipeline

print("Loading model... (This may take a while on first run)")
sentiment_pipeline = pipeline("sentiment-analysis")

sentences = [
    "This assignment is very easy and fun!",
    "The weather is terrible today, I don't want to do anything.",
    "Artificial intelligence is the technology of the future."
]

print("\n--- Results ---")
for sentence in sentences:
    result = sentiment_pipeline(sentence)[0]
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']} (Score: {result['score']:.4f})\n")
```

### OUTPUT
Using a pipeline without specifying a model name and revision in production is not recommended.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████| 104/104 [00:00<00:00, 10582.94it/s]

--- Results ---
Sentence: This assignment is very easy and fun!
Sentiment: POSITIVE (Score: 0.9999)

Sentence: The weather is terrible today, I don't want to do anything.
Sentiment: NEGATIVE (Score: 0.9996)

Sentence: Artificial intelligence is the technology of the future.
Sentiment: POSITIVE (Score: 0.9949)


### Why This Model?
- DistilBERT is 60% faster than BERT but keeps 97% of its accuracy.
- It is small enough to run on a normal computer without a GPU.
- It is already fine-tuned for sentiment analysis, so no training is needed.

### Advantages of Hugging Face
- Very easy to use. One line of code loads a model.
- Many pre-trained models are available for free.
- Good documentation and community support.
- Works on CPU and GPU.

### Dataset Structure
- Training Data: SST-2 dataset (movie reviews with positive/negative labels).
- Input for Inference: Any English text.
- Output: A label (POSITIVE/NEGATIVE) and a confidence score (0 to 1).

### Evaluation Metrics
- Accuracy: Not calculated because the model was not re-trained.
- Confidence Score: The model gives a score between 0 and 1 for each prediction.
- Observation: All predictions had very high confidence (above 0.99).


### Task 2, (2_audio.py): Audio Processing (Speech-to-Text)

### Library Used
- Hugging Face Transformers (`transformers`)
- SoundFile (`soundfile`) for reading `.wav` files.

### Model Used
- `facebook/wav2vec2-base-960h`
- This model is trained on 960 hours of English speech.
- It converts audio into text.

### How the Model Works 
1. The audio wave is converted into small feature vectors.
2. A convolutional layer extracts important patterns.
3. A Transformer encoder processes the sequence.
4. A final layer outputs character probabilities.
5. The text is decoded using CTC (Connectionist Temporal Classification).

### CODE
```python
from transformers import pipeline
import soundfile as sf

print("Loading model...")
speech_recognizer = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")

audio_file_path = "test_audio_fixed.wav"

try:
    audio, sample_rate = sf.read(audio_file_path)
    print(f"Audio loaded successfully. Sample rate: {sample_rate}")

    result = speech_recognizer(audio)
    print("\n--- Transcription Result ---")
    print(f"Recognized Text: {result['text']}")
    
except Exception as e:
    print(f"Error occurred: {e}")
    print("Please make sure 'test_audio.wav' is in the project folder.")
```

### OUTPUT
Loading model...
Could not reach the Hub ([Errno 8] nodename nor servname provided, or not known). Using cached commit hash for 'facebook/wav2vec2-base-960h'.
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 212/212 [00:00<00:00, 8908.67it/s]
[transformers] Wav2Vec2ForCTC LOAD REPORT from: facebook/wav2vec2-base-960h
Key                        | Status  | 
---------------------------+---------+-
wav2vec2.masked_spec_embed | MISSING | 

 Notes:
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
Audio loaded successfully. Sample rate: 16000

--- Transcription Result ---
Recognized Text: THE WORLDY IS YOURS


### Why This Model?
- Wav2Vec2 is a state-of-the-art model for speech recognition.
- It is pre-trained on a large dataset (LibriSpeech).
- It works well even with different accents and background noise.

### Advantages of Hugging Face for Audio
- Pre-trained speech models are free and easy to use.
- The pipeline API handles complex audio processing.
- Supports different audio formats.
- Works on CPU (no GPU needed for small files).

### Dataset Structure
- Training Data: LibriSpeech (960 hours of English speech).
- Input for Inference: .wav audio files.
- Output: Transcribed text.

### Evaluation Metrics
- Word Error Rate (WER): Not calculated (no re-training).
- Output: The model successfully converted audio to text.
- Observation: The result depends on audio quality. Clear speech gives better results.


### Task 3, (3_image.py): Image Processing (Image Classification)

### Library Used
- PyTorch (`torch`, `torchvision`)
- Pillow (`PIL`) for loading images.

### Model Used
- `ResNet50` (Residual Network with 50 layers)
- Pre-trained on ImageNet (1000 object classes).
- ResNet50 uses "residual connections" to avoid the vanishing gradient problem.

### How the Model Works
1. The image is resized to 256x256 and center-cropped to 224x224.
2. Pixel values are normalized using ImageNet mean and standard deviation.
3. The image passes through 50 convolutional layers.
4. Residual connections skip some layers to help the model learn deep features.
5. The final fully-connected layer outputs 1000 probabilities (one for each class).
6. The top 3 predictions are displayed with their confidence scores.

### CODE
```python
import torch
from torchvision import models, transforms
from PIL import Image

print("Loading model...")
model = models.resnet50(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img = Image.open("test_image.jpg")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

with torch.no_grad():
    out = model(batch_t)

probabilities = torch.nn.functional.softmax(out[0], dim=0)
top3_prob, top3_catid = torch.topk(probabilities, 3)

print("\n--- Prediction Results ---")
for i in range(top3_prob.size(0)):
    print(f"Class ID: {top3_catid[i].item()}, Probability: {top3_prob[i].item():.4f}")
```

### OUTPUT
--- Prediction Results ---
Class ID: 281, Probability: 0.5249
Class ID: 282, Probability: 0.1578
Class ID: 283, Probability: 0.1518

### Why This Model?
- ResNet50 is a classic and reliable model for image classification.
- It is pre-trained on ImageNet, which has 1.2 million images.
- It works well on a normal computer without a GPU.

### Advantages of PyTorch
- Dynamic computation graph makes debugging easier.
- Torchvision provides many pre-trained models.
- Large community and good documentation.
- Works well on both CPU and GPU.

### Dataset Structure
- Training Data: ImageNet (1.2 million images, 1000 classes).
- Input for Inference: .jpg or .png images.
- Output: Class ID and probability for top 3 predictions.

### Evaluation Metrics
- Accuracy: Not calculated (no re-training).
- Confidence Score: The model gives a probability for each class (0 to 1).
- Observation: The model correctly identified the image as a cat with high confidence.


### Task 4, (4_video.py): Video Processing (Object Detection)

### Library Used
- Hugging Face Transformers (`transformers`)
- OpenCV (`cv2`) for reading video frames.
- Pillow (`PIL`) for image conversion.

### Model Used
- `facebook/detr-resnet-50`
- DETR = DEtection TRansformer.
- It combines a CNN backbone (ResNet-50) with a Transformer encoder-decoder.
- It detects objects in images without needing anchor boxes or NMS (Non-Maximum Suppression).

### How the Model Works
1. The video is read frame by frame using OpenCV.
2. Each frame is converted from BGR (OpenCV format) to RGB (PIL format).
3. The DETR model takes the image and extracts features using ResNet-50.
4. A Transformer encoder-decoder predicts bounding boxes and class labels.
5. The model outputs detected objects with confidence scores.
6. Only the first 10 frames are processed to save time and computing power.

### CODE 
```python
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
```

### OUTPUT
Loading model...
Loading weights: 100%|███████████████████████████████████████████████████████████████████████████████████████| 530/530 [00:00<00:00, 7589.30it/s]
[transformers] DetrForObjectDetection LOAD REPORT from: facebook/detr-resnet-50
Key                                                            | Status     |  | 
---------------------------------------------------------------+------------+--+-
model.backbone.model.layer3.0.downsample.1.num_batches_tracked | UNEXPECTED |  | 
model.backbone.model.layer4.0.downsample.1.num_batches_tracked | UNEXPECTED |  | 
model.backbone.model.layer1.0.downsample.1.num_batches_tracked | UNEXPECTED |  | 
model.backbone.model.layer2.0.downsample.1.num_batches_tracked | UNEXPECTED |  | 
Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Processing video... (First 10 frames)

Frame 1:
  - Object: giraffe, Score: 0.96

Frame 2:
  - Object: giraffe, Score: 1.00

Frame 3:
  - Object: giraffe, Score: 1.00

Frame 4:
  - Object: giraffe, Score: 1.00

Frame 5:
  - Object: giraffe, Score: 1.00

Frame 6:
  - Object: giraffe, Score: 1.00

Frame 7:
  - Object: giraffe, Score: 1.00

Frame 8:
  - Object: giraffe, Score: 1.00

Frame 9:
  - Object: giraffe, Score: 0.69
  - Object: giraffe, Score: 0.99

Frame 10:
  - Object: giraffe, Score: 1.00
Processing complete.

### Why This Model?
- DETR is a modern and accurate model for object detection.
- It does not need anchor boxes or NMS, which makes it simpler than YOLO.
- It works well with Hugging Face's pipeline API.

### Advantages of Hugging Face for Video
- Same API for images and video frames (easy to use).
- DETR is a modern and accurate model.
- Easy to combine with OpenCV for video processing.
- No need for anchor boxes or NMS (simpler than YOLO).

### Dataset Structure
- Training Data: COCO dataset (330K images, 80 object categories).
- Input for Inference: .mp4 video files.
- Output: Detected objects with confidence scores per frame.

### Evaluation Metrics
- Mean Average Precision (mAP): Not calculated (no re-training).
- Confidence Score: The model gives a probability for each detected object.
- Observation: The model detected the giraffe with very high confidence (0.96 - 1.00) in almost every frame.


## Task 5: LLM (Large Language Model) - NOT COMPLETED

### Status
Could not be completed because of download problems.

### What Was Tried
- Library: `llama-cpp-python`
- Model: `TheBloke/Llama-2-7B-Chat-GGUF` (4-bit quantized, ~4.5 GB)

### Errors Encountered
- `RuntimeError: Task error: File reconstruction error`
- `curl: (92) HTTP/2 stream 1 was not closed cleanly`

These errors mean the internet connection was not stable during the large download.

### Alternative Plan (Not Done)
A smaller model called `TinyLlama-1.1B-Chat` (~600 MB) was considered but not downloaded because of time limits.