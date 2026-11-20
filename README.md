# SpamShield - SMS Spam Detection

A small NLP project that tells whether an SMS message is **Spam** or **Not Spam (Ham)**.
Built with Python, scikit-learn and Streamlit.

## What it does
- Cleans and preprocesses SMS text (lowercase, remove punctuation, stopwords, stemming)
- Converts text to numbers using TF-IDF (unigrams + bigrams)
- Trains and compares Naive Bayes, Logistic Regression and Linear SVM
- Saves the best model and shows predictions with confidence in a Streamlit app

## Dataset
SMS Spam Collection (UCI), 5,574 labelled messages. 403 duplicates were removed.
File: `data/sms.tsv`

## How to run
```bash
pip install -r requirements.txt
python train.py          # trains the models, makes graphs in images/, saves model.pkl
streamlit run app.py     # opens the web app
```

## Results (test set)
See `metrics.json`. Best model: Linear SVM, accuracy about 98.8%, F1 (spam) about 0.95.

## Files
- `preprocess.py` - text cleaning functions
- `train.py` - full training and evaluation pipeline
- `app.py` - Streamlit GUI
- `images/` - EDA graphs and confusion matrix (add GUI screenshots here too)

## Team
Tejas Raghav (2401730187), Nishant Shokeen (2401730224), Soumya Jena (2401730228)
