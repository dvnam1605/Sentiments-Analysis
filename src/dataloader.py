import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from . import config

def load_and_prepare_data():
   
    df = pd.read_csv(config.DATA_PATH)

    if config.TEXT_COLUMN not in df.columns or config.LABEL_COLUMN not in df.columns:
        raise ValueError(f"CSV phải có các cột '{config.TEXT_COLUMN}' và '{config.LABEL_COLUMN}'")
        
    df.dropna(subset=[config.TEXT_COLUMN, config.LABEL_COLUMN], inplace=True)
    
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    train_val_df, test_df = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=42,
        stratify=df['label']
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.VALIDATION_SIZE / (1 - config.TEST_SIZE), # Điều chỉnh tỷ lệ
        random_state=42,
        stratify=train_val_df['label']
    )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })

    print("Tải và chuẩn bị dữ liệu hoàn tất:")
    print(dataset_dict)
    
    return dataset_dict