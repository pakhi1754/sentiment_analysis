from transformers import AutoModelForSequenceClassification

def load_model(num_labels):
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    return model