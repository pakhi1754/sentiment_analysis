from transformers import Trainer, TrainingArguments
from data_preprocessing import load_data, tokenize_data
from model import load_model
from evaluation import compute_metrics

def main():
    print("Loading datatset...")
    ds = load_data()

    print("Tokenizing...")
    tokenized_dataset, tokenizer = tokenize_data(ds)
    
    print("Loading model...")
    num_labels = 3
    model = load_model(num_labels)

    print("Setting training arguments...")
    training_args = TrainingArguments(
        output_dir="./results", 
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=1e-4,
        load_best_model_at_end=True
    )
    
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics
    )
    
    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_dataset["test"])
    print("\nFinal Test Results:")
    print(test_results)

    model.save_pretrained("./saved_model")
    tokenizer.save_pretrained("./saved_model")

if __name__ == "__main__":
    main()