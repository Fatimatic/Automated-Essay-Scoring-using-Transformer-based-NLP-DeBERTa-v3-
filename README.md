# Automated-Essay-Scoring-using-Transformer-based-NLP-DeBERTa-v3-
Built an automated essay scoring system using DeBERTa v3 and transformer-based NLP, predicting 6 analytic writing rubrics with low error using MCRMSE evaluation.


# Automated Essay Scoring using DeBERTa v3 (Transformer-based NLP)

## 📌 Project Overview
Manual essay grading is time-consuming, subjective, and often biased toward essay length rather than writing quality.  
This project presents an **Automated Essay Scoring (AES)** system using **transformer-based NLP** to evaluate essays consistently and fairly across multiple writing dimensions.

I built a deep learning model using **DeBERTa v3 Small**, a state-of-the-art transformer architecture, to predict **six analytic writing rubrics** simultaneously.

---

## 🎯 Problem Statement
Human graders:
- May unintentionally favor longer essays
- Can be inconsistent due to fatigue or bias
- Struggle to score large volumes efficiently

The goal was to **automate essay scoring** while focusing on **quality, structure, and language proficiency** rather than content length.

---

## 📊 Dataset
- **Dataset Name:** ELLIPSE Corpus (Kaggle)
- **Essay Type:** Argumentative essays
- **Authors:** Grade 8–12 English Language Learners (ELLs)
- **Scoring Rubrics (Targets):**
  - Cohesion
  - Syntax
  - Vocabulary
  - Phraseology
  - Grammar
  - Conventions
- **Score Range:** 1.0 – 5.0 (increments of 0.5)

⚠️ The full dataset is **not included** due to size and redistribution restrictions.  
A small sample dataset is provided for demonstration purposes.

---


Raw Essay Text
↓
Text Cleaning & Preparation
↓
Baseline (TF-IDF) Experiment
↓
Tokenizer (DeBERTa Tokenizer)
↓
Transformer Encoder (DeBERTa v3 Small)
↓
CLS Token Embedding
↓
Linear Regression Head
↓
Six Rubric Score Predictions



---

## 🧠 Why Not TF-IDF?
TF-IDF treats text as isolated word frequencies and **ignores context, grammar, and sentence structure**.  
Essay quality depends on **semantic coherence and syntax**, which require contextual understanding.

Hence,I moved to **transformer-based contextual embeddings**.

---

## 🏗 Model Architecture

- **Backbone:** DeBERTa v3 Small (Transformer Encoder)
- **Embedding Size:** 768
- **Pooling Strategy:** CLS token representation
- **Head:** Fully connected linear layer (regression)
- **Dropout:** 0.2 (to reduce overfitting)
- **Output:** 6 continuous values (one per rubric)

---

## ⚙️ Training Configuration

| Parameter | Value |
|--------|------|
| Optimizer | AdamW |
| Loss Function | Mean Squared Error (MSE) |
| Evaluation Metric | MCRMSE |
| Epochs | 1 (overfitting observed beyond this) |
| Batch Size | 4 |
| Max Token Length | 128 |
| Device | CPU |

> Training on CPU took approximately **1.5 hours**.

---

## 📐 Evaluation Metric
I used **Mean Column-wise Root Mean Squared Error (MCRMSE)**, which evaluates prediction error independently for each rubric and then averages them.

R² was not used due to **low variance in rubric scores**.

---

## 🚀 Inference (User Input Prediction)
The system allows users to paste an essay and receive predicted rubric scores instantly.

Example output:
Cohesion: 3.50
Syntax: 3.00
Vocabulary: 3.50
Phraseology: 3.00
Grammar: 4.00
Conventions: 4.50



---

## 📦 Trained Model
The trained model file (~530MB) is hosted externally due to GitHub size limits.

👉 **Download model weights here:**  
https://drive.google.com/file/d/1SktU7SaviJmV4Hybuy6XdtrVRdBMs4vj/view?usp=sharing

After downloading, place the file in the project root directory before running inference.

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Train the Model
python training.py

### 3️⃣ Run Inference
python inference.py


## 🔁 Project Workflow

🔮 Future Improvements

Train on GPU for longer context lengths

Use larger DeBERTa variants

Incorporate sentence-level attention analysis

Ensemble multiple transformer models

🧑‍💻 Skills Demonstrated

Transformer-based NLP

Regression modeling

Deep learning optimization

Evaluation metric design

End-to-end ML pipeline

Real-world inference deployment

📚 Technologies Used

Python

PyTorch

Hugging Face Transformers

DeBERTa v3

Scikit-learn

Pandas & NumPy

⭐ Summary

This project demonstrates an end-to-end NLP system that combines modern transformer architectures with educational assessment, highlighting practical machine learning skills under real-world constraints.
