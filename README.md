# sample-Analysis-chatbot

# Sentiment Classifier Chatbot

A Streamlit-powered chatbot that uses a pretrained transformer (DistilBERT) to classify the sentiment of user input text (positive or negative). The project demonstrates how to build, fine-tune, and deploy a sentiment analysis model using Hugging Face Transformers and the IMDB dataset, with interactive inference via a web app.

---

## 🚀 Features

- **Pretrained Transformer Model:** Uses `distilbert-base-uncased` from Hugging Face.
- **Binary Sentiment Classification:** Predicts if input text is positive or negative.
- **Interactive Web App:** Built with Streamlit for easy, real-time sentiment analysis.
- **Model Fine-tuning:** Optionally fine-tune the model on IMDB data.
- **Basic Evaluation Metrics:** Accuracy, precision, recall, and F1-score.
- **Attention Visualization:** (Bonus) Visualizes attention weights for sample predictions.
- **Extensible:** Easily swap model/dataset or add more features.

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Alisha-z/sentiment-classifier-chatbot.git
   cd sentiment-classifier-chatbot
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or install manually
   pip install streamlit torch transformers datasets scikit-learn matplotlib seaborn
   ```

---

## 💻 Usage

### 1. **Train or Fine-Tune Model**

You can fine-tune the model using the provided script:

```bash
python sentiment_classifier.py
```

This will:
- Load IMDB dataset
- Preprocess and tokenize data
- Fine-tune DistilBERT
- Evaluate model performance

### 2. **Run the Streamlit Chatbot App**

```bash
streamlit run sentiment_app.py
```

- Enter your sentence in the input box.
- Click "Classify" to get the sentiment prediction.

---

## 🧠 Project Structure

```text
sentiment-classifier-chatbot/
├── sentiment_classifier.py    # Main script: model training, evaluation, inference
├── sentiment_app.py           # Streamlit app for interactive sentiment classification
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (this file)
```

---

## 📝 Example

- **Input:** `I absolutely loved this movie!`
- **Output:** `Sentiment: Positive`

---

## 🎨 Bonus: Attention Visualization

You can visualize attention weights for a sample text by calling the `visualize_attention()` function in `sentiment_classifier.py`.

---

## ⚡️ Customization

- Swap for another transformer model (e.g., BERT, RoBERTa) by changing model name in the scripts.
- Use a different dataset (e.g., SST-2) by modifying the data loading section.

---

## 🤝 Contributing

Pull requests and suggestions are welcome!  
Feel free to open an issue for bug reports or feature requests.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙋‍♀️ Author

**Alisha-z**



- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
