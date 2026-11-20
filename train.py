"""
SpamShield - training script
Run:  python train.py
Does: data cleaning -> preprocessing -> EDA plots -> TF-IDF -> train 3 models
      -> evaluate -> save the best model in model.pkl
"""
import json
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)
from preprocess import preprocess

# ---------------- 1. load data ----------------
df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "message"],
                 quoting=3, on_bad_lines="skip", encoding="utf-8")
print("Raw shape:", df.shape)

# ---------------- 2. data cleaning ----------------
print("Missing values:\n", df.isnull().sum())
df = df.dropna()
dups = df.duplicated().sum()
print("Duplicate rows:", dups)
df = df.drop_duplicates().reset_index(drop=True)
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["ham", "spam"])]
print("Shape after cleaning:", df.shape)

# ---------------- 3. preprocessing ----------------
df["clean"] = df["message"].apply(preprocess)
df["target"] = (df["label"] == "spam").astype(int)
df["length"] = df["message"].str.len()
df["words"] = df["message"].str.split().str.len()

# ---------------- 4. EDA ----------------
counts = df["label"].value_counts()
plt.figure(figsize=(5, 4))
plt.bar(counts.index, counts.values, color=["#4c72b0", "#c44e52"])
plt.title("Ham vs Spam count")
plt.ylabel("Messages")
plt.tight_layout(); plt.savefig("images/class_distribution.png", dpi=120); plt.close()

plt.figure(figsize=(6, 4))
plt.hist(df[df.label == "ham"]["length"], bins=40, alpha=0.7, label="ham", color="#4c72b0")
plt.hist(df[df.label == "spam"]["length"], bins=40, alpha=0.7, label="spam", color="#c44e52")
plt.title("Message length (characters)")
plt.xlabel("Length"); plt.legend()
plt.tight_layout(); plt.savefig("images/length_distribution.png", dpi=120); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, lab, col in zip(ax, ["ham", "spam"], ["#4c72b0", "#c44e52"]):
    words = " ".join(df[df.label == lab]["clean"]).split()
    top = Counter(words).most_common(10)
    a.barh([w for w, _ in top][::-1], [c for _, c in top][::-1], color=col)
    a.set_title("Top words in " + lab)
plt.tight_layout(); plt.savefig("images/top_words.png", dpi=120); plt.close()

# ---------------- 5. split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["target"], test_size=0.2, random_state=42, stratify=df["target"])

# ---------------- 6. models ----------------
models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5),
}

results, fitted = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
    }
    fitted[name] = (pipe, pred)
    print("\n==", name, "==")
    print(classification_report(y_test, pred, target_names=["ham", "spam"]))

# ---------------- 7. pick best by F1 ----------------
best = max(results, key=lambda k: results[k]["f1"])
print("Best model:", best, results[best])
best_pipe, best_pred = fitted[best]

ConfusionMatrixDisplay(confusion_matrix(y_test, best_pred), display_labels=["ham", "spam"]).plot(cmap="Blues")
plt.title("Confusion matrix - " + best)
plt.tight_layout(); plt.savefig("images/confusion_matrix.png", dpi=120); plt.close()

joblib.dump(best_pipe, "model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"best_model": best, "results": results,
               "train_size": int(len(X_train)), "test_size": int(len(X_test)),
               "duplicates_removed": int(dups)}, f, indent=2)
print("Saved model.pkl and metrics.json")
