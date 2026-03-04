from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score


def run_tfidf_baseline(train_df, test_df):
    """
    TF-IDF + Logistic Regression baseline.
    param train_df: training dataframe
    param test_df: test dataframe
    returns: sentiment and toxicity f1 scores
    """
    print("\n" + "="*55)
    print(" BASELINE: TF-IDF + LOGISTIC REGRESSION")
    print("="*55)

    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=2, strip_accents="unicode", analyzer="word")

    X_train = tfidf.fit_transform(train_df["text"])
    X_test  = tfidf.transform(test_df["text"])

    # Sentiment
    print("\nTraining sentiment classifier...")
    lr_sent = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight={0: 2.0, 1: 1.0}, random_state=42,)
    lr_sent.fit(X_train, train_df["sentiment_label"])
    sent_preds = lr_sent.predict(X_test)

    sent_f1  = f1_score(test_df["sentiment_label"], sent_preds, average="macro")
    print(f"\n[TF-IDF Sentiment] Macro F1: {sent_f1:.4f}")
    print(classification_report(test_df["sentiment_label"], sent_preds, target_names=["negative", "positive"], digits=4))

    # Toxicity
    print("Training toxicity classifier...")
    lr_tox = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", class_weight="balanced", random_state=42,)
    lr_tox.fit(X_train, train_df["toxicity_label"])
    tox_preds = lr_tox.predict(X_test)

    tox_f1  = f1_score(test_df["toxicity_label"], tox_preds, average="macro")
    print(f"\n[TF-IDF Toxicity] Macro F1: {tox_f1:.4f}")
    print(classification_report(test_df["toxicity_label"], tox_preds, target_names=["counter", "neutral", "hate"], digits=4))

    return sent_f1, tox_f1