from transformers import AutoTokenizer, AutoModelForSequenceClassification
from . import config

def load_model_and_tokenizer():
    """Tải PhoBERT model và tokenizer từ Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=len(config.LABEL_MAP),
        id2label=config.ID2LABEL,
        label2id=config.LABEL_MAP
    )
    
    print(f"Đã tải xong model '{config.MODEL_NAME}' và tokenizer.")
    return model, tokenizer