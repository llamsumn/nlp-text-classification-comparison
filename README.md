# NLP Text Classification: BOW vs Semantic Approach

A comparative study of two NLP pipelines for text classification, implemented as part of coursework for CM52065.

---

## 📌 Overview

This project explores and compares two distinct approaches to classifying text data:

| Approach | Preprocessing | Vectorisation | Classifier |
|---|---|---|---|
| **BOW** | Stemming (Porter) + TF-IDF | Char n-gram TF-IDF (2–4) | LinearSVC |
| **Semantic** | Lemmatisation (spaCy) | Sentence Transformers (`all-MiniLM-L6-v2`) | Random Forest |

Both pipelines handle class imbalance, preserve negations during stop word removal, and are evaluated using macro F1-score with 5-fold cross-validation.

---

## 📁 Repository Structure

```
├── BOW_Approach.ipynb        # Bag-of-Words pipeline
├── Semantic_Approach.ipynb   # Semantic embedding pipeline
├── cw_data.csv               # Dataset (text + label columns)
└── README.md
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install datasets scikit-learn nltk sentence-transformers spacy matplotlib
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

1. Clone the repo and navigate to the project folder
2. Ensure `cw_data.csv` is in the same directory as the notebooks
3. Open and run either notebook:
   - `BOW_Approach.ipynb` — for the TF-IDF + LinearSVC pipeline
   - `Semantic_Approach.ipynb` — for the Sentence Transformer + Random Forest pipeline

---

## 📊 Evaluation

Both models are evaluated using:
- Classification report (precision, recall, F1 per class)
- Confusion matrix
- 5-fold cross-validated macro F1-score

---

## 📝 Notes

- Negation words (`not`, `no`, `cannot`) are intentionally retained during stop word removal to preserve sentiment signals
- The BOW approach uses character-level n-grams (`char_wb`, 2–4) for robustness to morphological variation
- The Semantic approach leverages pre-trained sentence embeddings for richer contextual representation
