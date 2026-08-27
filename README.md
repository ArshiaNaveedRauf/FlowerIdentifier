# Flower Recognition
A convolutional neural network, built from scratch with TensorFlow/Keras, that classifies photos of flowers into one of five species: daisy, dandelion, rose, sunflower, or tulip. Includes a small Flask app for uploading a photo and getting a live prediction.

## Results

~72% validation accuracy. Daisy, dandelion, sunflower, and tulip are all identified reliably; rose is the weakest class, most often confused with tulip. See the full report for details.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download the [Kaggle Flowers Recognition dataset](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) and place it at `data/raw/flowers/`, with one subfolder per class.

## Usage

**Train a new model:**
```bash
python3 main.py
```

**Evaluate the already-saved model** (fast, no retraining):
```bash
python3 saved_model_pipeline.py
```

## Testing

```bash
pytest
```

CI runs this automatically on every push via GitHub Actions (`.github/workflows/ci.yml`).
