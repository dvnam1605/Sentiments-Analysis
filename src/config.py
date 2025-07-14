import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'output.csv')
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, 'models', 'phobert-sentiment')

TEXT_COLUMN = 'text'   # Tên cột chứa văn bản
LABEL_COLUMN = 'label' # Tên cột chứa nhãn
LABEL_MAP = {
    'NEGATIVE': 0,
    'NEUTRAL': 1,
    'POSITIVE': 2
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

# Tỷ lệ chia dữ liệu
TEST_SIZE = 0.1
VALIDATION_SIZE = 0.1

# --- CẤU HÌNH MODEL ---
MODEL_NAME = "vinai/phobert-base"
MAX_LENGTH = 128 # Độ dài tối đa của một câu sau khi tokenize

# --- CẤU HÌNH HUẤN LUYỆN ---
TRAINING_ARGS = {
    "output_dir": MODEL_OUTPUT_DIR,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "num_train_epochs": 10,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 150,
    "weight_decay": 0.01,
    "logging_dir": os.path.join(BASE_DIR, 'logs'),
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "report_to": "none" 
}