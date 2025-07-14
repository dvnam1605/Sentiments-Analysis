import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments

from . import config
from . import data_loader
from . import model

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    return {'accuracy': acc, 'f1': f1}

def main():
    dataset_dict = data_loader.load_and_prepare_data()

    phobert_model, tokenizer = model.load_model_and_tokenizer()

    def preprocess_function(examples):
        return tokenizer(
            examples[config.TEXT_COLUMN], 
            truncation=True, 
            padding="max_length", 
            max_length=config.MAX_LENGTH
        )

    # 4. Áp dụng tokenization lên toàn bộ dataset
    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)
    print("\nTokenization hoàn tất.")

    training_args = TrainingArguments(**config.TRAINING_ARGS)

    trainer = Trainer(
        model=phobert_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    test_results = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
    print(test_results)

    trainer.save_model(config.MODEL_OUTPUT_DIR)
    print(f"\nModel và tokenizer đã được lưu tại: {config.MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    main()