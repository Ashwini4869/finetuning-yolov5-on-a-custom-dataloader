# Finetuning YOLOv5 on a custom dataset

## Environment Setup
Create a virtual environment using `python3 -m venv venv` and activate it:

```bash
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Set the `hyperparameters`, `model_paths` and `dataset_paths` in `src/config.cfg`.

Run the training script:

```bash
python src/train.py
```


