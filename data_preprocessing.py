from datasets import load_dataset
from sklearn.model_selection import train_test_split
from config import RANDOM_SEED, TOX_MIN, TOX_MAX, SENT_THRESHOLD, TOX_THRESHOLD_LOW, TOX_THRESHOLD_HIGH


def load_raw_data():
    """ 
    Function for loading dataframe
    returns: the loaded dataframe
    """
    dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")
    df = dataset["train"].to_pandas()
    print(f"Raw shape: {df.shape}")
    print(f"Unique posts: {df['comment_id'].nunique()}")
    print(f"Unique annotators: {df['annotator_id'].nunique()}")
    return df


def aggregate_annotations(df):
    """
    Function for averaging the scores by multiple annotators for the same post.
    param df: the raw dataframe
    returns: the aggregated dataframe
    """
    agg_df = df.groupby("comment_id").agg(
        text=("text", "first"),
        sentiment=("sentiment", "mean"),
        hate_speech_score=("hate_speech_score", "mean"),
        insult=("insult", "mean"),
        humiliate=("humiliate", "mean"),
        dehumanize=("dehumanize", "mean"),
        violence=("violence", "mean"),
    ).reset_index()
    print(f"\nAfter aggregation: {agg_df.shape}")
    return agg_df


def normalize_and_label(agg_df):
    """
    Function for normalizing the data nd creating labels.
    param df: the aggregated dataframe
    returns: the dataframe with normalized scores and labels
    """
    agg_df["toxicity_score"]  = (agg_df["hate_speech_score"] - TOX_MIN) / (TOX_MAX - TOX_MIN)

    for col in ["sentiment", "insult", "humiliate", "dehumanize", "violence"]:
        agg_df[col] = agg_df[col] / 4.0

    agg_df["sentiment_label"] = (agg_df["sentiment"] >= SENT_THRESHOLD).astype(int)
    agg_df["toxicity_label"]  = agg_df["toxicity_score"].apply(lambda s: 0 if s < TOX_THRESHOLD_LOW else (1 if s < TOX_THRESHOLD_HIGH else 2))

    print("\nSentiment class distribution:")
    print(agg_df["sentiment_label"].value_counts().sort_index().rename({0: "negative", 1: "positive"}))
    print("\nToxicity class distribution:")
    print(agg_df["toxicity_label"].value_counts().sort_index().rename({0: "counter", 1: "neutral", 2: "hate"}))

    return agg_df


def split_data(agg_df):
    """80/10/10 stratified split on toxicity label."""
    train_df, temp_df = train_test_split(
        agg_df, test_size=0.2, random_state=RANDOM_SEED,
        stratify=agg_df["toxicity_label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=RANDOM_SEED,
        stratify=temp_df["toxicity_label"]
    )
    print(f"\nSplit sizes — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df