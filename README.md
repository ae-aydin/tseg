# Tumor Segmentation for QuPath

## Workflow

1. _qupath-scripts_: Create tiles from annotated slides on QuPath.

2. _dataset-preparation_: Prepare dataset from all created tiles.

3. _model-training_: Train a segmentation model.

## Setup

```python
# uv, one of those
uv sync
uv pip install -r requirements.txt
uv pip install -r pyproject.toml

# pip
pip install -r requirements.txt
```

## References

1. [NoCodeSeg Repository by andreped](https://github.com/andreped/NoCodeSeg)

2. [Rectlabel-support by ryouchinsa](https://github.com/ryouchinsa/Rectlabel-support)
