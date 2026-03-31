import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D,
    Input, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
DATA_DIR   = "screenshots"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# Keras assigns class indices alphabetically.
# genuine_site_0 → 0  |  phishing_site_1 → 1
CLASSES = ['genuine_site_0', 'phishing_site_1']


# =========================
# DATA GENERATORS
# =========================
# Genuine (majority, 1447 images) — mild augmentation
genuine_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)

# Phishing (minority, 557 images) — aggressive augmentation
# to compensate for fewer samples
phishing_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    shear_range=0.1,
    channel_shift_range=20.0,
    validation_split=0.2,
)

# Combined generator for validation (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

# Single combined train generator (uses mild augmentation for both)
# The class_weight parameter handles the imbalance during training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    shear_range=0.08,
    channel_shift_range=15.0,
    validation_split=0.2,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    classes=CLASSES,
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    classes=CLASSES,
    shuffle=False,  # Must be False for correct evaluation
    seed=SEED,
)

print("\nClass mapping:", train_generator.class_indices)
print("Expected → {'genuine_site_0': 0, 'phishing_site_1': 1}")
print("Training samples :", train_generator.samples)
print("Validation samples:", val_generator.samples)


# =========================
# CLASS WEIGHTS
# =========================
labels  = train_generator.classes
counter = Counter(labels)
print("\nTraining class distribution:", dict(counter))
# genuine=0, phishing=1 — expect ~1157 genuine, ~445 phishing (80% split)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels,
)
class_weights = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights (auto-balanced):", class_weights)
# phishing weight will be ~2.6x genuine weight


# =========================
# MODEL
# =========================
def build_model(trainable_base=False, unfreeze_from=None):
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = trainable_base
    if trainable_base and unfreeze_from is not None:
        for layer in base_model.layers[:unfreeze_from]:
            layer.trainable = False

    inputs = Input(shape=(224, 224, 3))

    # training=False → BN layers stay in inference mode during phase 1
    # This is critical for stable transfer learning
    x = base_model(inputs, training=not trainable_base)
    x = GlobalAveragePooling2D()(x)

    # Stronger head — original single Dense(1) was too weak
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs), base_model


model, base_model = build_model(trainable_base=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

model.summary()


# =========================
# CALLBACKS
# =========================
os.makedirs("models", exist_ok=True)

def get_callbacks(prefix):
    return [
        ModelCheckpoint(
            filepath=f"models/best_{prefix}.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=7,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_auc",
            patience=3,
            factor=0.3,
            min_lr=1e-7,
            mode="max",
            verbose=1,
        ),
    ]


# =========================
# PHASE 1 — Frozen base
# Train only the new classification head
# =========================
print("\n" + "="*50)
print("PHASE 1 — Training head (base frozen)")
print("="*50)

history1 = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=get_callbacks("phase1"),
)


# =========================
# PHASE 2 — Fine-tuning
# Unfreeze top ~97 layers of EfficientNetB0 (~237 total)
# Original code only unfroze after layer 50 — way too few
# =========================
print("\n" + "="*50)
print("PHASE 2 — Fine-tuning (top layers unfrozen)")
print("="*50)

base_model.trainable = True

# Freeze bottom 140 layers, unfreeze top ~97
for layer in base_model.layers[:140]:
    layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Trainable base layers: {trainable_count} / {len(base_model.layers)}")

# Much lower LR to avoid destroying pretrained ImageNet weights
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

history2 = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=get_callbacks("phase2"),
)


# =========================
# THRESHOLD CALIBRATION
# Default 0.5 is almost never optimal on imbalanced data.
# Sweep thresholds and pick the one that maximises F1.
# =========================
print("\n" + "="*50)
print("THRESHOLD CALIBRATION")
print("="*50)

val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=1)
y_true = val_generator.classes[: len(y_pred_probs)]

auc_score = roc_auc_score(y_true, y_pred_probs)
print(f"Validation AUC: {auc_score:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)

f1_scores = []
for t in thresholds:
    preds = (y_pred_probs >= t).astype(int).flatten()
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    f1_scores.append(f1)

best_idx       = int(np.argmax(f1_scores))
best_threshold = float(thresholds[best_idx])
best_f1        = float(f1_scores[best_idx])
print(f"Optimal threshold : {best_threshold:.4f}")
print(f"Best F1 score     : {best_f1:.4f}")


# =========================
# FINAL EVALUATION
# =========================
print("\n" + "="*50)
print("FINAL EVALUATION (optimal threshold)")
print("="*50)

y_pred_labels = (y_pred_probs >= best_threshold).astype(int).flatten()

print(classification_report(
    y_true,
    y_pred_labels,
    target_names=["genuine", "phishing"],
))

cm = confusion_matrix(y_true, y_pred_labels)
print("Confusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")


# =========================
# SAVE MODEL + THRESHOLD
# =========================
model.save("models/phishing_model_final.keras")

with open("models/threshold.json", "w") as f:
    json.dump({"threshold": best_threshold}, f, indent=2)

print(f"\n✅ Model saved    → models/phishing_model_final.keras")
print(f"✅ Threshold saved → models/threshold.json  ({best_threshold:.4f})")


# =========================
# TRAINING CURVES
# =========================
def plot_history(h1, h2, metric, title, fname):
    t1 = h1.history.get(metric, [])
    v1 = h1.history.get(f"val_{metric}", [])
    t2 = h2.history.get(metric, [])
    v2 = h2.history.get(f"val_{metric}", [])

    ep1 = list(range(1, len(t1) + 1))
    ep2 = list(range(len(t1) + 1, len(t1) + len(t2) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(ep1, t1, "b-",  label="Train P1")
    plt.plot(ep1, v1, "b--", label="Val P1")
    plt.plot(ep2, t2, "r-",  label="Train P2 (fine-tune)")
    plt.plot(ep2, v2, "r--", label="Val P2 (fine-tune)")
    plt.axvline(len(t1), color="gray", linestyle=":", alpha=0.7, label="Fine-tune start")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.upper())
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"models/{fname}.png", dpi=150)
    plt.close()
    print(f"Plot saved → models/{fname}.png")

plot_history(history1, history2, "auc",      "Validation AUC",      "auc_curve")
plot_history(history1, history2, "loss",     "Loss",                 "loss_curve")
plot_history(history1, history2, "accuracy", "Accuracy",             "acc_curve")
plot_history(history1, history2, "recall",   "Recall (phishing det)","recall_curve")

print("\n✅ All plots saved to models/")


# =========================
# INFERENCE HELPER
# Use this in app.py to predict a single screenshot
# =========================
def predict_image(img_path, model_path="models/phishing_model_final.keras",
                  threshold_path="models/threshold.json"):
    """
    Predict whether a screenshot is phishing or genuine.

    Returns:
        label      : "phishing" or "genuine"
        confidence : float probability of being phishing (0.0 – 1.0)

    Usage in app.py:
        from imageintegrate import predict_image
        label, conf = predict_image("path/to/screenshot.png")
    """
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image

    with open(threshold_path) as f:
        threshold = json.load(f)["threshold"]

    loaded_model = load_model(model_path)

    img = keras_image.load_img(img_path, target_size=IMAGE_SIZE)
    arr = keras_image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob  = float(loaded_model.predict(arr, verbose=0)[0][0])
    label = "phishing" if prob >= threshold else "genuine"
    return label, prob


# =========================
# QUICK SMOKE TEST
# Uncomment and set a real path to verify after training
# =========================
# label, prob = predict_image("screenshots/phishing_site_1/some_image.png")
# print(f"Result: {label}  |  P(phishing) = {prob:.4f}")