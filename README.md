# 📈 Financial Sentiment Analysis

An NLP deep learning model that classifies financial news sentences as **Positive**, **Neutral**, or **Negative**.

## 📌 Problem
Financial news moves markets, and analysts can't read everything. Automatically detecting the sentiment of financial text helps with market monitoring, investment research, and risk alerts. The task is to classify each sentence into one of three sentiment classes.

## 📊 Dataset
- **[Financial Sentiment Analysis (Kaggle)](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis)**: financial news sentences, each labeled with a sentiment
- **5,836 sentences** after removing duplicates
- The dataset is imbalanced: Neutral 3,124 · Positive 1,852 · Negative 860

## 🧠 Approach
1. **EDA:** checked for missing values and duplicates, plotted the class distribution, and generated a **word cloud for each sentiment**.
2. **Text preprocessing:** a custom cleaning function that removes mentions, hashtags, URLs, punctuation, and symbols, followed by tokenization (NLTK), lowercasing, stop-word removal, and **lemmatization**.
3. **Handling imbalance:** random oversampling of the minority classes.
4. **Feature engineering:** Keras `Tokenizer` (with an out-of-vocabulary token) and padded sequences.
5. **Model:** a stacked **LSTM** network: Embedding (100-dim) → LSTM(128) → LSTM(64) → dense layers with Dropout → softmax over 3 classes.
6. **Training:** Adam optimizer (lr = 0.001) with **early stopping** (patience = 6).
7. **Evaluation:** accuracy, a per-class classification report, and a confusion matrix.


## 🔍 Try it
The notebook includes a `prediction()` function for classifying new sentences:
```python
prediction(["The agreement will provide The Switch with double the converter capacity"])
# Predicted Class is : positive
```

## 🚀 How to run
```bash
git clone https://github.com/youstinashraf/Financial-Sentiment-Analysis.git
cd Financial-Sentiment-Analysis
pip install -r requirements.txt
```
1. Download `data.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis).
2. Open `Financial_Sentiment_Analysis.ipynb` in Jupyter or Google Colab and run all cells.

## 🛠️ Tech stack
Python · TensorFlow / Keras (LSTM) · NLTK · scikit-learn · imbalanced-learn · Pandas · NumPy · Matplotlib / Seaborn · WordCloud

## 🔮 Future improvements
- Fine-tune a transformer model built for finance (e.g., **FinBERT**)
- Use pre-trained word embeddings (GloVe) in place of a randomly initialized embedding
- Deploy as a simple web app (Streamlit or Gradio)
