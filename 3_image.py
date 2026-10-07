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