# Melanoma Detection: GWO-ANN

Phân loại ung thư da Melanoma sử dụng **Grey Wolf Optimizer + Artificial Neural Network**.

## Cấu trúc

```
├── main.py              # Chạy chính
├── data_loader.py       # Load & tiền xử lý dữ liệu
├── ann_model.py         # Model ANN
├── gwo_optimizer.py     # Grey Wolf Optimizer
├── gwo_ann_trainer.py   # Huấn luyện GWO-ANN & Standard ANN
└── melanoma_cancer_dataset/
    ├── train/           # 9605 ảnh
    └── test/            # 1000 ảnh
```

## Tiền xử lý

1. Resize ảnh → 64×64
2. Normalize (ImageNet)
3. Flatten → 12,288D
4. PCA → 100D
5. Standardization

## Model

```
Input(100) → Dense(64) → Sigmoid → Dropout(0.3) → Dense(2) → Output
```

## Chạy

```bash
# Cài đặt
pip install -r requirements.txt

# Huấn luyện cả 2 model
python main.py

# Chỉ Standard ANN
python main.py --standard

# Chỉ GWO-ANN
python main.py --gwo
```

## Kết quả

So sánh **GWO-ANN** vs **Standard ANN** trên các metrics: Accuracy, Precision, Recall, F1, Sensitivity, Specificity.

