# Vietnamese Sentiment Analysis with PhoBERT

A Vietnamese sentiment analysis project using PhoBERT (Vietnamese BERT) for classifying text into positive, negative, or neutral sentiments.

## 📋 Overview

This project implements a sentiment analysis model specifically designed for Vietnamese text using the PhoBERT pre-trained model from VinAI. The model can classify Vietnamese text into three categories:

- **POSITIVE**: Positive sentiment
- **NEGATIVE**: Negative sentiment
- **NEUTRAL**: Neutral sentiment

## 🏗️ Project Structure

```
sentiment/
├── data/
│   └── merged_output.csv          # Training dataset
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── dataloader.py              # Data loading and preprocessing
│   ├── model.py                   # Model loading utilities
│   ├── predict.py                 # Prediction utilities
│   └── train.py                   # Training script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/dvnam1605/Sentiments-Analysis.git
cd Sentiments-Analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Training the Model

1. Prepare your data in CSV format with columns `text` and `label`
2. Update the data path in `src/config.py` if needed
3. Run training:

```bash
python -m src.train
```

### Making Predictions

```python
from src.predict import predict_manually

# Example usage
text = "thầy dạy rất nhiệt tình và dễ hiểu"
model_path = "path/to/your/trained/model"
result = predict_manually(text, model_path)
print(f"Predicted: {result['label']} (confidence: {result['score']:.4f})")
```

## 📊 Model Configuration

The model uses the following key configurations:

- **Base Model**: `vinai/phobert-base`
- **Max Sequence Length**: 128 tokens
- **Training Epochs**: 10
- **Batch Size**: 16
- **Learning Rate**: Adaptive with warmup
- **Evaluation Metric**: F1-score (weighted)

## 📝 Data Format

Your training data should be in CSV format with the following columns:

```csv
text,label
"Sản phẩm này rất tốt",POSITIVE
"Chất lượng không như mong đợi",NEGATIVE
"Bình thường, không có gì đặc biệt",NEUTRAL
```

## 🔧 Configuration

All configuration settings are centralized in `src/config.py`:

- **Data paths**: Input data and model output directories
- **Model settings**: PhoBERT model name, sequence length
- **Training parameters**: Epochs, batch size, learning rate, etc.
- **Label mapping**: Sentiment categories and their numeric IDs

## 📈 Training Process

The training process includes:

1. **Data Loading**: Load and preprocess CSV data
2. **Data Splitting**: Train/validation/test split (80%/10%/10%)
3. **Tokenization**: Convert text to PhoBERT tokens
4. **Training**: Fine-tune PhoBERT on your sentiment data
5. **Evaluation**: Calculate accuracy and F1-score metrics
6. **Model Saving**: Save the best model based on F1-score

## 🎯 Performance Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Weighted F1-score across all classes
- **Per-class metrics**: Precision, recall, and F1 for each sentiment

## 💡 Usage Examples

### Basic Prediction

```python
from src.predict import predict_manually

# Positive example
result1 = predict_manually("Dịch vụ tuyệt vời!", "path/to/model")
# Output: {'label': 'POSITIVE', 'score': 0.9521}

# Negative example
result2 = predict_manually("Rất thất vọng về sản phẩm", "path/to/model")
# Output: {'label': 'NEGATIVE', 'score': 0.8745}
```

### Batch Prediction

For multiple texts, you can modify the prediction function or call it in a loop.

## 🛠️ Development

### Adding New Features

1. Update configuration in `src/config.py`
2. Modify data loading in `src/dataloader.py`
3. Adjust model architecture in `src/model.py`
4. Update training logic in `src/train.py`

### Custom Labels

To use different sentiment categories:

1. Update `LABEL_MAP` in `src/config.py`
2. Ensure your data uses the new labels
3. Retrain the model

## 📋 Requirements

- `transformers[torch]`: Hugging Face Transformers with PyTorch
- `accelerate`: Training acceleration
- `datasets`: Dataset handling
- `scikit-learn`: Evaluation metrics
- `pandas`: Data manipulation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VinAI Research** for the PhoBERT model
- **Hugging Face** for the Transformers library
- The Vietnamese NLP community for resources and datasets

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size in `config.py`
2. **Model loading errors**: Check the model path in prediction scripts
3. **Data format issues**: Ensure CSV has correct column names (`text`, `label`)

### Getting Help

If you encounter issues:

1. Check the error messages carefully
2. Verify your data format matches the expected structure
3. Ensure all dependencies are installed correctly
4. Check that model paths are correct

## 📊 Example Results

When trained on educational feedback data, the model achieved:

- **Training Loss**: 0.013
- **Evaluation Loss**: 0.32
- **F1-Score**: 0.94
- **Training Time**: ~0.5 hours (depending on dataset size and hardware)

These results demonstrate excellent performance with very low training loss and high F1-score, indicating the model has learned to classify Vietnamese sentiment effectively.

---

**Note**: Remember to update the model path in `src/predict.py` before running predictions with your trained model.
