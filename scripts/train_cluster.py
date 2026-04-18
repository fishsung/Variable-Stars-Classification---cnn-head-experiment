import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import pandas as pd
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 256

def prepare_hybrid_data(subset_name):
    base_path = f"final_subclass_split/{subset_name}"
    csv_dir = "datasets"

    csv_files = [f for f in os.listdir(csv_dir) if f.startswith(subset_name) and f.endswith(".csv")]
    df_list = [pd.read_csv(os.path.join(csv_dir, f)) for f in csv_files]
    master_df = pd.concat(df_list, ignore_index=True).set_index('id')
    
    master_df = master_df.dropna(subset=['period', 'windex'])

    file_paths, labels, periods, w_indexes = [], [], [], []
    class_names = sorted(os.listdir(base_path))
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    print(f"Syncing {subset_name} data...")
    for class_name in class_names:
        class_dir = os.path.join(base_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            star_id = img_name.replace(".png", "")
            if star_id in master_df.index:
                file_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_to_idx[class_name])
                periods.append(master_df.loc[star_id, 'period'])
                w_indexes.append(master_df.loc[star_id, 'windex'])

    def load_and_preprocess(path, period, w_index, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = img / 255.0
        label = tf.one_hot(label, 16)
        return {"image_input": img, "period_input": [period], "w_input": [w_index]}, label

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, periods, w_indexes, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset, len(file_paths), class_names, labels  # raw integer labels returned for eval


full_train_ds, total_train_samples, class_names, _ = prepare_hybrid_data("train")
test_ds, _, _, test_labels_raw = prepare_hybrid_data("test")

val_size = int(0.2 * total_train_samples)

full_train_ds = full_train_ds.shuffle(buffer_size=total_train_samples, seed=123)
val_ds  = full_train_ds.take(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
train_ds = full_train_ds.skip(val_size).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_variable_star_cnn(input_shape=(128, 128, 1), num_classes=16, dropout_rate=0.3):
    image_input = layers.Input(shape=input_shape, name="image_input")

    # Block 1 — low-level feature extraction + resize
    x = layers.Conv2D(32, (7, 7), activation="relu", padding="valid", name="conv1_7x7")(image_input)
    x = layers.Conv2D(32, (5, 5), activation="relu", padding="valid", name="conv2_5x5")(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 — high-level features
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv3_3x3")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv4_3x3")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="valid", name="conv5_3x3")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 — high-level features (mirror of block 2)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="valid", name="conv6_3x3")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="valid", name="conv7_3x3")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="valid", name="conv8_3x3")(x)
    x = layers.Dropout(dropout_rate, name="dropout3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 — final conv block before flatten
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="valid", name="conv9_3x3")(x)
    x = layers.Conv2D(256, (3, 3), activation="relu", padding="valid", name="conv10_3x3")(x)
    x = layers.Dropout(dropout_rate, name="dropout4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # FC layers for image branch
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, activation="relu", name="cnn_dense_256")(x)
    x = layers.Dense(64, activation="relu", name="cnn_dense_64")(x)
    cnn_features = layers.Dense(16, activation="softmax", name="cnn_dense_16")(x)

    # Period branch — single Dense+softmax as in paper
    period_input = layers.Input(shape=(1,), name="period_input")
    period_features = layers.Dense(16, activation="softmax", name="period_dense_16")(period_input)

    # Wesenheit index branch — single Dense+softmax as in paper
    w_input = layers.Input(shape=(1,), name="w_input")
    w_features = layers.Dense(16, activation="softmax", name="w_dense_16")(w_input)

    # Concatenate all three branches and classify
    merged = layers.concatenate([cnn_features, period_features, w_features], name="concat")
    x = layers.Dense(64, activation="relu", name="merged_dense_64")(merged)
    output = layers.Dense(num_classes, activation="softmax", name="final_output")(x)

    return models.Model(inputs=[image_input, period_input, w_input], outputs=output)


def get_training_callbacks(checkpoint_path="best_hybrid_model.weights.h5"):
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=19,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
    ]

# ── Build & train ──────────────────────────────────────────────────────────────
my_model = build_variable_star_cnn(num_classes=16)
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2.5e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
my_model.summary()

print("GPUs available: ", len(tf.config.list_physical_devices('GPU')))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=500,
    callbacks=get_training_callbacks(),
    verbose=1
)

# ── Save model weights ─────────────────────────────────────────────────────────
my_model.save_weights("final_16class_model.weights.h5")
print("Saved final weights → final_16class_model.weights.h5")

# ── Evaluate on test set ───────────────────────────────────────────────────────
test_loss, test_acc = my_model.evaluate(test_ds, verbose=1)
print(f"\nTest loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.4f}")

# ── Confusion matrix ───────────────────────────────────────────────────────────
print("\nGenerating confusion matrix...")
y_pred_probs = my_model.predict(test_ds, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(test_labels_raw)

cm = confusion_matrix(y_true, y_pred)

# Normalise rows to get per-class recall (matches paper's confusion matrix style)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

# Plot — normalised confusion matrix
fig_cm, ax_cm = plt.subplots(figsize=(14, 12))
im = ax_cm.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

ax_cm.set_xticks(range(len(class_names)))
ax_cm.set_yticks(range(len(class_names)))
ax_cm.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
ax_cm.set_yticklabels(class_names, fontsize=9)
ax_cm.set_xlabel("Predicted label", fontsize=11)
ax_cm.set_ylabel("True label", fontsize=11)

# Annotate cells with percentage value
thresh = 0.5
for i in range(len(class_names)):
    for j in range(len(class_names)):
        val = cm_norm[i, j]
        ax_cm.text(j, i, f"{val:.2f}",
                   ha="center", va="center", fontsize=7,
                   color="white" if val > thresh else "black")

plt.tight_layout()
fig_cm.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix → confusion_matrix.png")

# ── Save history as JSON ───────────────────────────────────────────────────────
hist_dict = history.history
hist_dict["test_loss"] = float(test_loss)
hist_dict["test_accuracy"] = float(test_acc)
hist_dict["confusion_matrix"] = cm.tolist()          # raw counts
hist_dict["confusion_matrix_norm"] = cm_norm.tolist() # normalised
hist_dict["class_names"] = class_names

with open("training_history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)
print("Saved history → training_history.json")

# ── Plot training curves ───────────────────────────────────────────────────────
epochs_range = range(1, len(hist_dict["accuracy"]) + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(epochs_range, hist_dict["accuracy"],   label="Train accuracy")
axes[0].plot(epochs_range, hist_dict["val_accuracy"], label="Val accuracy")
axes[0].set_title("Accuracy over epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(epochs_range, hist_dict["loss"],     label="Train loss")
axes[1].plot(epochs_range, hist_dict["val_loss"], label="Val loss")
axes[1].set_title("Loss over epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

fig.suptitle("16-class MINN — Szklenár et al. 2022 reproduction", fontsize=13)
plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
print("Saved plot → training_curves.png")
