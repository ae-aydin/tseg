# Model Training for Tumor Segmentation

This repository contains the scripts necessary to train, evaluate, and export segmentation models using the `segmentation_models_pytorch` library.

---

## 1. Training (`train.py`)

Used to train models using standard splits or cross-validation.

**Single Fold:**
```bash
python train.py config.yaml
```

**Cross-Validation:**
```bash
python train.py config.yaml --cv
```

---

## 2. Validation (`val.py`)

Performs detailed performance analysis, including ROC/PR curves and threshold optimization.

**Run Analysis:**
```bash
python val.py path/to/model/ [config.yaml] --split val
```

---

## 3. Testing (`test.py`)

Evaluates the model on the test set and saves prediction samples.

**Run Test:**
```bash
python test.py path/to/model/ [config.yaml]
```

---

## 4. Conversion (`convert.py`)

Converts trained PyTorch models to ONNX and TorchScript formats for deployment.

**Convert:**
```bash
python convert.py path/to/model/ path/to/output/ model_name
```
