"""Shared constants. Everything that training and inference must agree on lives here."""

DENSITY_CLASSES = ["A", "B", "C", "D"]
DENSITY_DESCRIPTIONS = {
    "A": "Almost entirely fatty",
    "B": "Scattered areas of fibroglandular density",
    "C": "Heterogeneously dense (may obscure small masses)",
    "D": "Extremely dense (lowers sensitivity of mammography)",
}
VIEWS = ["CC", "MLO"]
MALIGNANCY_CLASSES = ["benign", "malignant"]

# Preprocessed (cached) breast crops are stored at this height; width follows the aspect ratio.
CACHE_HEIGHT = 1024
# Network input size (H, W). Mammograms are tall, so we keep a portrait canvas.
INPUT_H = 768
INPUT_W = 512

# Ignore-label used for samples that lack a target for a given head.
IGNORE = -100
