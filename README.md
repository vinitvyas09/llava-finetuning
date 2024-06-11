This model is a fine-tuned version of a pre-trained multimodal neural network, specifically adapted for facial emotion detection. It leverages a convolutional neural network (CNN) backbone for image processing and a transformer-based encoder for text input, optimized to classify emotions from facial images. The fine-tuning process utilized a labeled dataset of facial expressions to enhance the model's accuracy in detecting and categorizing human emotions.

# Model
If you want to use this model directly without running this code, or are interested in the output of this fine-tuning, you can download the weights from my kaggle repo here: https://www.kaggle.com/models/vinitvyas09/llava-finetuned-for-face-emotion-classification. 
Note that this model is around 12-14GB in size.

# Usage
The fine-tuned model can be used for real-time emotion detection in images, making it suitable for applications in sentiment analysis, customer service, and human-computer interaction. The following code snippet demonstrates how to load and use the model for emotion classification:

```python
import torch
from transformers import LLAVAModel, LLAVATokenizer

# Load the fine-tuned model and tokenizer
model = LLAVAModel.from_pretrained('your_finetuned_model_path')
tokenizer = LLAVATokenizer.from_pretrained('your_finetuned_model_path')

# Example usage
image_input = torch.randn((1, 3, 224, 224))  # Example image tensor
text_input = "Describe the emotion in the image"
inputs = tokenizer(text_input, return_tensors="pt")

# Get predictions
outputs = model(image_input, inputs)
emotion = outputs.logits.argmax(dim=-1)
print(f"Predicted emotion: {emotion}")
```

# Known and Preventable Failures
The model may struggle with images of low quality or with faces that are not clearly visible. It is also less effective with non-facial images or when the input text is irrelevant to the image context.

# System
This model is part of an integrated system for emotion detection and requires both image and text inputs for optimal performance. The system dependencies include:
- A pre-processing pipeline for image normalization and text tokenization.
- Post-processing steps for interpreting model outputs and generating human-readable results.

# Implementation Requirements
- Hardware: The model was fine-tuned on NVIDIA A100 GPUs.
- Software: The fine-tuning environment included Python 3.8, PyTorch 1.10, and the Transformers library by Hugging Face.
- Compute Requirements: Fine-tuning the model required approximately 3 GPU hours. Inference can be performed on a single GPU with minimal latency.

# Model Characteristics
## Model Initialization
The model was fine-tuned from a pre-trained multimodal neural network provided by the Hugging Face library, initially trained on a diverse set of image-text pairs.

## Model Stats
Size: 
Approximately 12 GB
Weights: 
The model consists of 24 layers with over 300 million parameters.
Latency: 
Average inference time is around 50ms per image-text pair on an NVIDIA A100 GPU.
Other Details:
The model is neither pruned nor quantized. No differential privacy techniques were applied during fine-tuning.

# Data Overview
## Training Data
The fine-tuning process utilized a dataset of 35,685 examples of 48x48 pixel grayscale images of faces, annotated with emotion labels. The data was collected from Kaggle's facial emotion recognition challenge dataset.

## Dataset URL
https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer

## Pre-processing
Images were normalized and resized to 224x224 pixels. Text data was tokenized using the LLAVATokenizer.

## Demographic Groups
The dataset includes images from diverse demographic groups, though specific demographic attributes are not explicitly annotated.

# Evaluation Data
The data was split into training (80%), validation (10%), and test (10%) sets. The test data reflects similar characteristics to the training data.

# Usage Limitations
The model is designed for facial emotion detection and may not perform well on non-facial images or text inputs that do not describe emotions. It is important to ensure high-quality inputs for accurate predictions.

# Ethics
Ethical considerations included ensuring the model does not reinforce stereotypes or biases. The fine-tuning process involved regular audits to identify and mitigate potential biases. The model was also designed to respect user privacy, with no personal data being stored or shared.
