# Deep Learning Projects

## **Overview**
This repository contains deep learning projects using state-of-the-art neural networks, pretrained models, and transfer learning across vision, NLP, and audio domains. These projects are designed to demonstrate the power of deep neural networks in real-world classification tasks.

---

## **Project Structure**

```plaintext
/
├── README.md                                 # Main documentation
├── Speaker Identification/                   # Speaker recognition using deep audio features
├── Sentiment_Classification_BERT.ipynb       # NLP sentiment classification using BERT
├── ImageClassification/                      # Image classification using CNN + Transfer Learning

```
## **Key Features**

- **Domains Covered:** Speech Recognition, Natural Language Processing, and Computer Vision.
- **Model Categories:**
  - Deep audio classification (Speaker Recognition)
  - Transformer-based models (BERT for NLP)
  - Convolutional Neural Networks (Image Classification)
- **Libraries and Tools:** PyTorch, torchaudio, HuggingFace Transformers, TensorFlow/Keras, Matplotlib, seaborn, NumPy, pandas.
- **Workflow Steps:**
  - Problem definition
  - Data preprocessing and augmentation
  - Feature extraction
  - Model training and evaluation
  - Visualization and interpretation

---

### **1. Speaker Recognition and Classification**

#### **Summary**
This project identifies speakers using audio recordings by leveraging pretrained ECAPA-TDNN embeddings from the SpeechBrain library. It explores both classification and similarity-based approaches to speaker recognition, and includes a Gradio demo app for interactive testing. Designed using a small sample dataset, this project showcases a lightweight hybrid speaker recognition pipeline.

#### **Highlights**
- **Data:** Small-scale speaker dataset with short audio clips representing multiple identities.
- **Methods:**
  - **Preprocessing:** Mono conversion, resampling to 16 kHz, and fixed-length waveform padding using `torchaudio`.
  - **Embedding Extraction:** ECAPA-TDNN model from `speechbrain.pretrained.EncoderClassifier`.
  - **Two Approaches:**
    -  **Classification-based**: A custom neural network (`EmbeddingClassifierBN`) trained on embeddings for multi-class speaker identification.
    -  **Similarity-based**: Cosine similarity between embeddings to verify if two audio clips are from the same speaker.
  - **Hybrid Evaluation:** Combines both approaches for flexible inference and robustness testing.
- **Interface:** A user-friendly **Gradio app** allows real-time testing of speaker audio to evaluate both classification and similarity predictions.
- **Evaluation:** Accuracy, training loss curves, confusion matrices, and real-world inference via the Gradio interface.

📁 **Project Directory:** [Speaker Recognition](./Speaker%20Identification/)

---

### 🎬 Sentiment Classification with BERT (IMDB Movie Reviews)

#### 📝 **Project Summary**
This project leverages **BERT** (Bidirectional Encoder Representations from Transformers) for fine-tuned sentiment classification on the **IMDB movie review dataset**. The goal is to predict the sentiment— **positive**, or **negative**—of a given movie review using Transformer-based contextual embeddings.

#### 🚀 **Project Highlights**
- **📚 Dataset:** IMDB movie reviews labeled with sentiment classes.
- **🔧 Methods & Tools:**
  - Pretrained BERT model: `bert-base-uncased` 
  - Input preprocessing: tokenization, padding, attention masks
  - Fine-tuning BERT on the labeled sentiment data
- **📊 Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, F1-score
  - Confusion Matrix for class-wise performance

#### 📁 **Project Notebook**
 [Sentiment Classification with BERT (via Transfer Learning)](./Sentiment_Classification_with_BERT_via_Transfer_Learning.ipynb)

---

### **3. Image Classification via Transfer Learning**

#### **Summary**
This computer vision project uses transfer learning with pretrained CNNs (like ResNet or VGG) for classifying images into different categories. It reduces training time while achieving strong performance.

#### **Highlights**
- **Data:** A folder-based image dataset with labels.
- **Methods:**
  - Pretrained CNN (e.g., ResNet18) from `torchvision.models`
  - Data augmentation using transforms
  - Fine-tuning last layers on custom dataset
- **Evaluation:** Training accuracy, validation accuracy, confusion matrix, and visualizations.

📁 **Project Directory:** [Image Classification](./ImageClassification/)

---

## **Contributing**

Contributions are welcome! Fork the repository, make your changes, and submit a pull request. For major changes, please open an issue to discuss what you’d like to change.

---
