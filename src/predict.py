import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def predict_manually(text_to_analyze: str, model_path: str):

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Lỗi khi tải model hoặc tokenizer: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    # Sử dụng tokenizer để chuyển văn bản thành định dạng model yêu cầu
    inputs = tokenizer(
        text_to_analyze, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    predicted_class_id = torch.argmax(logits, dim=1).item()
  
    predicted_label = model.config.id2label[predicted_class_id]
    
    score = probabilities[predicted_class_id]

    result = {
        'label': predicted_label,
        'score': float(score)
    }
    
    print("\n--- KẾT QUẢ ---")
    print(f"Nhãn dự đoán: {result['label']}")
    print(f"Điểm tin cậy: {result['score']:.4f}")
    
    return result

if __name__ == "__main__":
    model_dir = "path/to/your/model"  # Thay đổi đường dẫn tới model đã huấn luyện
    
    # Các câu để thử nghiệm
    text1 = "thầy dạy rất nhiệt tình và dễ hiểu"
    text2 = "giáo trình còn sơ sài, cần cập nhật thêm"
    
    predict_manually(text1, model_dir)
    print("\n" + "="*50 + "\n")
    predict_manually(text2, model_dir)