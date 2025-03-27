# **Deepfake-vs-Real-8000**  

> **Deepfake-vs-Real-8000** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to detect whether an image is a deepfake or a real one using the **SiglipForImageClassification** architecture.  


```py
Classification Report:
              precision    recall  f1-score   support

    Deepfake     0.9990    0.9972    0.9981      4000
    Real one     0.9973    0.9990    0.9981      4000

    accuracy                         0.9981      8000
   macro avg     0.9981    0.9981    0.9981      8000
weighted avg     0.9981    0.9981    0.9981      8000
```

![download.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/MqyYUuGb-gZDsCtusIQOr.png)

The model categorizes images into two classes:  
- **Class 0:** "Deepfake"  
- **Class 1:** "Real one"  

---

# **Run with Transformers🤗**  

```python
!pip install -q transformers torch pillow gradio
```  

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Deepfake-vs-Real-8000"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def deepfake_classification(image):
    """Predicts whether an image is a Deepfake or Real."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "Deepfake", "1": "Real one"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=deepfake_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Deepfake vs. Real Image Classification",
    description="Upload an image to determine if it's a Deepfake or a Real one."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

---

# **Intended Use:**  

The **Deepfake-vs-Real-8000** model is designed to detect deepfake images from real ones. Potential use cases include:  

- **Deepfake Detection:** Assisting cybersecurity experts and forensic teams in detecting synthetic media.  
- **Media Verification:** Helping journalists and fact-checkers verify the authenticity of images.  
- **AI Ethics & Research:** Contributing to studies on AI-generated content detection.  
- **Social Media Moderation:** Enhancing tools to prevent misinformation and digital deception.
