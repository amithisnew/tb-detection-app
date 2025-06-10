import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras.saving
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
import time
from tqdm import tqdm
import traceback
import shutil
import subprocess

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = "1"

# Clear any existing TensorFlow session
tf.keras.backend.clear_session()

# --- Configuration ---
U_ZEROS_POLICY = True
FORCE_FRESH_TRAINING = False  # Resume from checkpoints
BATCH_SIZE = 8
EPOCHS = 25
LEARNING_RATE = 1e-5
CHECKPOINT_INTERVAL = 50
MAX_RETRIES = 3
VALIDATION_CHECKPOINT_INTERVAL = 1000

# --- Paths ---
ROOT = "/content/drive/MyDrive/TB_Detection_Project"
MODEL_DIR = os.path.join(ROOT, "models")
DRIVE_DATA_DIR = os.path.join(ROOT, "data", "preprocessed")
LOCAL_DATA_DIR = "/content/data/preprocessed"
INVALID_LOG_FILES = os.path.join(MODEL_DIR, "invalid_npy_files.txt")
TRAINING_LOG_PATH = os.path.join(MODEL_DIR, "training_log.txt")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "cbm_checkpoint_3")
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, "cbm_weights_3.weights.h5")
EPOCH_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cbm_epoch.txt")
BATCH_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "cbm_batch.txt")
VALIDATION_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "validation_checkpoint.npz")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.weights.h5")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.weights.h5")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "concept_predictions.npy")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)

# --- Write to log file ---
def write_log(message):
    with open(TRAINING_LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    print(message)

# --- Check disk space ---
def check_disk_space():
    result = subprocess.run(["df", "-h", "/content"], capture_output=True, text=True)
    write_log(f"Disk space check:\n{result.stdout}")
    free_space = int(subprocess.run(["df", "/content"], capture_output=True, text=True).stdout.splitlines()[-1].split()[3])
    if free_space < 5 * 1024 * 1024:  # Less than 5 GB in KB
        write_log("Warning: Low disk space on /content")
    return free_space > 1024 * 1024  # At least 1 GB

# --- Verify Google Drive mount ---
def verify_drive_mount():
    if not os.path.exists(DRIVE_DATA_DIR):
        write_log(f"Google Drive not mounted or DRIVE_DATA_DIR missing: {DRIVE_DATA_DIR}")
        return False
    write_log(f"Google Drive mounted. Checking files in {DRIVE_DATA_DIR}")
    file_count = len(glob.glob(os.path.join(DRIVE_DATA_DIR, "**/*.npy"), recursive=True))
    write_log(f"Found {file_count} .npy files in DRIVE_DATA_DIR")
    return file_count > 0

# --- Verify and copy data to local disk ---
def copy_data_to_local():
    print("Copying data to local disk...")
    start_time = time.time()
    if not check_disk_space():
        write_log("Insufficient disk space for data copy")
        return False
    if not verify_drive_mount():
        write_log("Drive mount verification failed")
        return False
    
    tb_files_expected = False
    for split in ["train", "valid", "test"]:
        for label in ["TB_Positive", "TB_Negative"]:
            src_dir = os.path.join(DRIVE_DATA_DIR, split, label)
            if os.path.exists(src_dir) and os.listdir(src_dir):
                tb_files_expected = True
                break
    if not tb_files_expected:
        write_log("Warning: No TB data found in DRIVE_DATA_DIR. Check data paths.")
    
    for attempt in range(MAX_RETRIES):
        try:
            shutil.rmtree(LOCAL_DATA_DIR, ignore_errors=True)
            os.makedirs(LOCAL_DATA_DIR)
            # Use rsync for robust copy
            cmd = ["rsync", "-ah", "--progress", DRIVE_DATA_DIR + "/", LOCAL_DATA_DIR + "/"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"rsync failed: {result.stderr}")
            print(f"Data copied in {time.time() - start_time:.2f} seconds")
            tb_count = sum(1 for split in ["train", "valid", "test"] for label in ["TB_Positive", "TB_Negative"]
                         for f in glob.glob(os.path.join(LOCAL_DATA_DIR, split, label, "*.npy")))
            if tb_count < 100:
                write_log(f"Warning: Only {tb_count} TB files copied. Expected more.")
            else:
                write_log(f"Copied {tb_count} TB files successfully.")
            return True
        except Exception as e:
            msg = f"Copy attempt {attempt + 1}/{MAX_RETRIES} failed: {e}"
            write_log(msg)
            time.sleep(1)
    write_log("Failed to copy data after max retries")
    return False

if not copy_data_to_local():
    raise RuntimeError("Data copy failed. Check DRIVE_DATA_DIR and disk space.")

# --- Verify checkpoint files ---
def verify_checkpoints():
    checkpoint_files = [WEIGHTS_PATH, EPOCH_CHECKPOINT_PATH, BATCH_CHECKPOINT_PATH, VALIDATION_CHECKPOINT_PATH]
    for f in checkpoint_files:
        if os.path.exists(f):
            write_log(f"Found checkpoint file: {f} ({os.path.getsize(f)/1024:.2f} KB)")
        else:
            write_log(f"Checkpoint file missing: {f}")
verify_checkpoints()

# --- Load CheXpert Dataset ---
try:
    image_paths = np.load(os.path.join(MODEL_DIR, "subset_image_paths.npy"), allow_pickle=True)
    labels = np.load(os.path.join(MODEL_DIR, "subset_labels.npy")).astype(np.float32)
    image_paths = [p.replace(DRIVE_DATA_DIR, LOCAL_DATA_DIR) for p in image_paths]
    print(f"Loaded {len(image_paths)} images and labels")
except Exception as e:
    msg = f"Failed to load dataset: {e}"
    write_log(msg)
    raise

# --- Load TB Datasets ---
tb_image_paths = []
for split in ["train", "valid", "test"]:
    for label in ["TB_Positive", "TB_Negative"]:
        path = os.path.join(LOCAL_DATA_DIR, split, label, "*.npy")
        files = glob.glob(path)
        print(f"Found {len(files)} files in {path}")
        tb_image_paths.extend(files)
tb_image_paths = np.array(tb_image_paths)
print(f"Loaded {len(tb_image_paths)} TB images")
if len(tb_image_paths) < 100:
    write_log("Warning: Limited TB data. Check DRIVE_DATA_DIR for missing files.")

# --- Use CheXpert for Training/Validation ---
valid_image_paths = image_paths
valid_labels = labels
valid_tb_paths = tb_image_paths
print(f"Validated {len(valid_image_paths)} images")

# Handle uncertain labels
valid_labels = np.where(valid_labels == -1, 0 if U_ZEROS_POLICY else 1, valid_labels)
valid_labels = np.clip(valid_labels, 0, 1)

# Compute class weights
class_weights = {}
pos_counts = np.sum(valid_labels, axis=0)
neg_counts = len(valid_labels) - pos_counts
for i in range(valid_labels.shape[1]):
    class_weights[i] = (neg_counts[i] / pos_counts[i]) * 1.5 if pos_counts[i] > 0 else 1.0
class_weights[2] *= 2.0  # Edema
class_weights[3] *= 2.0  # Atelectasis
print(f"Class weights: {class_weights}")

# Dataset split
train_paths, val_paths, train_labels, val_labels = train_test_split(
    valid_image_paths, valid_labels, test_size=0.1, random_state=42
)

# Filter invalid .npy files
invalid_files = set()
if os.path.exists(INVALID_LOG_FILES):
    with open(INVALID_LOG_FILES, "r") as f:
        invalid_files = set(line.split(":")[0].strip() for line in f.readlines())
invalid_files.update([
    "/content/data/preprocessed/023158.npy",
    "/content/data/preprocessed/001411.npy",
    "/content/data/preprocessed/024032.npy",
    "/content/data/preprocessed/018941.npy",
    "/content/data/preprocessed/007908.npy",
    "/content/data/preprocessed/010659.npy",
    "/content/data/preprocessed/017507.npy",
    "/content/data/preprocessed/025447.npy",
    "/content/data/preprocessed/024316.npy",
    "/content/data/preprocessed/004645.npy",
    "/content/data/preprocessed/023586.npy",
    "/content/data/preprocessed/000708.npy",
    "/content/data/preprocessed/002937.npy",
    "/content/data/preprocessed/022587.npy",
    "/content/data/preprocessed/006095.npy",
    "/content/data/preprocessed/014158.npy",
    "/content/data/preprocessed/008443.npy",
    "/content/data/preprocessed/003172.npy",
    "/content/data/preprocessed/002571.npy",
    "/content/data/preprocessed/003277.npy",
    "/content/data/preprocessed/025579.npy",
    "/content/data/preprocessed/017265.npy",
    "/content/data/preprocessed/005215.npy",
    "/content/data/preprocessed/015630.npy",
    "/content/data/preprocessed/018972.npy",
    "/content/data/preprocessed/014770.npy",
    "/content/data/preprocessed/005657.npy",
    "/content/data/preprocessed/006624.npy",
    "/content/data/preprocessed/009272.npy",
    "/content/data/preprocessed/012193.npy",
    "/content/data/preprocessed/025951.npy",
    "/content/data/preprocessed/010956.npy",
    "/content/data/preprocessed/012804.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/011748.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/004971.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/015720.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/019617.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/012893.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/018167.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/020606.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/021671.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/024194.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/018595.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/000478.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/023614.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/002111.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/023870.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/024429.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/005910.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/010360.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/013591.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/010445.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/011924.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/020448.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/016463.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/021080.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/012830.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/009331.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/003394.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/024960.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/000959.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/024716.npy",
    "/content/drive/MyDrive/TB_Detection_Project/processed/015989.npy",
])
train_paths_orig = train_paths.copy()
val_paths_orig = val_paths.copy()
train_paths = [p for p in train_paths if p not in invalid_files]
val_paths = [p for p in val_paths if p not in invalid_files]
train_labels = [l for p, l in zip(train_paths_orig, train_labels) if p not in invalid_files]
val_labels = [l for p, l in zip(val_paths_orig, val_labels) if p not in invalid_files]
print(f"Filtered {len(train_paths_orig) - len(train_paths)} invalid train files")
print(f"Filtered {len(val_paths_orig) - len(val_paths)} invalid val files")
print(f"Train paths: {len(train_paths)}, Validation paths: {len(val_paths)}")

# --- Validate train paths and shapes ---
valid_train_paths = []
valid_train_labels = []
start_index = 0
if os.path.exists(VALIDATION_CHECKPOINT_PATH):
    try:
        checkpoint = np.load(VALIDATION_CHECKPOINT_PATH, allow_pickle=True)
        valid_train_paths = checkpoint['paths'].tolist()
        valid_train_labels = checkpoint['labels'].tolist()
        start_index = len(valid_train_paths)
        print(f"Loaded validation checkpoint: {start_index} paths")
    except Exception as e:
        msg = f"Error loading validation checkpoint: {e}"
        write_log(msg)
        start_index = 0

for i, (p, l) in enumerate(tqdm(zip(train_paths[start_index:], train_labels[start_index:]), 
                                 total=len(train_paths) - start_index, 
                                 desc="Validating train paths")):
    if os.path.exists(p):
        try:
            img = np.load(p)
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Empty or invalid image")
            valid_train_paths.append(p)
            valid_train_labels.append(l)
        except Exception as e:
            invalid_files.add(p)
            with open(INVALID_LOG_FILES, "a") as f:
                f.write(f"{p}: Failed to load: {str(e)}\n")
    
    if (i + 1) % VALIDATION_CHECKPOINT_INTERVAL == 0 or (i + 1) == len(train_paths) - start_index:
        np.savez(VALIDATION_CHECKPOINT_PATH, 
                 paths=np.array(valid_train_paths), 
                 labels=np.array(valid_train_labels))
        msg = f"Validation checkpoint saved at {len(valid_train_paths)} paths"
        write_log(msg)

train_paths = valid_train_paths
train_labels = valid_train_labels
print(f"Validated train paths: {len(train_paths)}")
print(f"Expected batches per epoch: {len(train_paths) // BATCH_SIZE}")

# --- Optimized tf.data Pipeline ---
def load_image(path, label=None):
    def _load_npy(path):
        try:
            path = path.decode('utf-8') if isinstance(path, bytes) else path
            for attempt in range(MAX_RETRIES):
                try:
                    img = np.load(path)
                    break
                except Exception as e:
                    print(f"Load attempt {attempt + 1}/{MAX_RETRIES} failed for {path}: {e}")
                    time.sleep(1)
                    if attempt == MAX_RETRIES - 1:
                        raise
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                raise ValueError("Empty or invalid image")
            if len(img.shape) == 2:
                img = img[..., np.newaxis]
            elif len(img.shape) == 3:
                if img.shape[-1] > 3:
                    img = img[..., :1]
                elif img.shape[-1] == 1:
                    pass
                else:
                    img = img[..., :1]
            elif len(img.shape) == 4:
                if img.shape[-1] == 1:
                    img = img[..., 0, :1]
                elif img.shape[-2] == 1:
                    img = img[..., 0, :]
                    img = img[..., :1]
                else:
                    raise ValueError(f"Unsupported 4D shape {img.shape}")
            else:
                raise ValueError(f"Unsupported shape {img.shape}")
            img = tf.convert_to_tensor(img, dtype=tf.float32)
            img = tf.image.resize(img, [224, 224])
            img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img) + 1e-7)
            img = tf.repeat(img, 3, axis=-1)
            return tf.cast(img, tf.float32)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            write_log(msg)
            with open(INVALID_LOG_FILES, "a") as f:
                f.write(f"{path}: {str(e)}\n")
            return tf.zeros([224, 224, 3], dtype=tf.float32)
    
    img = tf.numpy_function(_load_npy, [path], tf.float32)
    img.set_shape([224, 224, 3])
    if label is not None:
        return img, label
    return img

def make_dataset(paths, labels=None, batch_size=BATCH_SIZE, shuffle=True, cache=False, skip_batches=0):
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=512, seed=42)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if skip_batches > 0:
            ds = ds.skip(skip_batches)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        if skip_batches > 0:
            ds = ds.skip(skip_batches)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

print("Creating datasets...")
start_time = time.time()
try:
    train_ds = make_dataset(train_paths, train_labels, cache=False)
    val_ds = make_dataset(val_paths, val_labels, shuffle=False, batch_size=4)
    tb_ds = make_dataset(valid_tb_paths, labels=None, shuffle=False) if len(valid_tb_paths) > 0 else None
    print(f"Datasets created in {time.time() - start_time:.2f} seconds")
except Exception as e:
    msg = f"Dataset creation failed: {e}"
    write_log(msg)
    raise

# --- CBM Model ---
def build_cbm(input_shape=(224, 224, 3), num_concepts=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomBrightness(0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    x = layers.RandomZoom(0.1)(x)
    base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_concepts, activation="sigmoid", dtype='float32')(x)
    return models.Model(inputs, outputs)

# --- Checkpoint Functions ---
def save_checkpoint(model, epoch, batch=0):
    model.save_weights(WEIGHTS_PATH)
    with open(EPOCH_CHECKPOINT_PATH, "w") as f:
        f.write(str(epoch))
    with open(BATCH_CHECKPOINT_PATH, "w") as f:
        f.write(str(batch))
    msg = f"Checkpoint saved at epoch {epoch}, batch {batch}"
    write_log(msg)

def load_checkpoint(model):
    if os.path.exists(WEIGHTS_PATH) and os.path.exists(EPOCH_CHECKPOINT_PATH):
        try:
            model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
            with open(EPOCH_CHECKPOINT_PATH, "r") as f:
                epoch = int(f.read().strip())
            batch = 0
            if os.path.exists(BATCH_CHECKPOINT_PATH):
                with open(BATCH_CHECKPOINT_PATH, "r") as f:
                    batch = int(f.read().strip())
            print(f"Loaded checkpoint: epoch {epoch}, batch {batch}")
            return epoch, batch
        except Exception as e:
            msg = f"Error loading {WEIGHTS_PATH}: {e}"
            write_log(msg)
    print("No valid checkpoint found. Starting from scratch.")
    return 0, 0

# --- Focal Loss ---
@keras.saving.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon())
        bce = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal_loss = alpha_t * tf.pow(1.0 - p_t, gamma) * bce
        return tf.reduce_mean(focal_loss)
    focal_loss_fixed.__name__ = 'focal_loss_fixed'
    return focal_loss_fixed

# --- Build + Compile ---
model = build_cbm()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True, name="auc")]
)

# Warm up model
print("Warming up model...")
with tqdm(total=1, desc="Model warmup") as pbar:
    dummy_input = tf.zeros((BATCH_SIZE, 224, 224, 3))
    model.predict(dummy_input, batch_size=BATCH_SIZE)
    pbar.update(1)
print("Model warmup complete")

initial_epoch, initial_batch = load_checkpoint(model)
if initial_epoch > 0 or initial_batch > 0:
    train_ds = make_dataset(train_paths, train_labels, cache=False, skip_batches=initial_batch)

# --- Callback for epoch end ---
class EpochCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.current_epoch = initial_epoch

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} completed. Loss: {logs.get('loss'):.4f}, Val AUC: {logs.get('val_auc'):.4f}")
        tf.keras.backend.clear_session()

# --- Checkpoint Callback ---
class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_callback):
        super().__init__()
        self.epoch_callback = epoch_callback

    def on_batch_end(self, batch, logs=None):
        if batch % CHECKPOINT_INTERVAL == 0:
            current_epoch = self.epoch_callback.current_epoch
            save_checkpoint(model, current_epoch, batch)

# --- Callbacks ---
epoch_callback = EpochCallback()
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=3, min_lr=1e-7),
    ModelCheckpoint(filepath=BEST_MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True, verbose=1),
    CheckpointCallback(epoch_callback),
    epoch_callback
]

# --- Train ---
class_weight_dict = {i: class_weights[i] for i in range(5)}
try:
    start_time = time.time()
    print(f"Starting training from epoch {initial_epoch}, batch {initial_batch}")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    print(f"Training complete. Final metrics: {history.history}")
    initial_epoch += len(history.history['loss'])
    with open(EPOCH_CHECKPOINT_PATH, "w") as f:
        f.write(str(initial_epoch))
    save_checkpoint(model, initial_epoch, 0)
    print(f"Training time: {(time.time() - start_time) / 3600:.2f} hours")
except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    save_checkpoint(model, epoch_callback.current_epoch, initial_batch)
    print(f"Checkpoint saved at epoch {epoch_callback.current_epoch}, batch {initial_batch}")
except Exception as e:
    msg = f"Training error: {e}\n{traceback.format_exc()}"
    write_log(msg)
    save_checkpoint(model, epoch_callback.current_epoch, initial_batch)
    print(f"Checkpoint saved at epoch {epoch_callback.current_epoch}, batch {initial_batch}")

# Clear memory
tf.keras.backend.clear_session()

# --- Evaluate ---
val_images, val_labels = [], []
for img, lbl in val_ds.unbatch().take(1000):
    val_images.append(img.numpy())
    val_labels.append(lbl.numpy())
val_images = np.array(val_images)
val_labels = np.array(val_labels)
try:
    y_pred = model.predict(val_images, batch_size=4)
    y_pred_binary = (y_pred > 0.5).astype(np.int32)
    concept_names = ["Effusion", "Consolidation", "Edema", "Atelectasis", "Lung Opacity"]
    print("\nClassification Report:")
    print(classification_report(val_labels, y_pred_binary, target_names=concept_names, zero_division=0))
    auc_scores = []
    for i in range(val_labels.shape[1]):
        if len(np.unique(val_labels[:, i])) > 1:
            auc = roc_auc_score(val_labels[:, i], y_pred[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(0.0)
            print(f"Warning: Only one class present in {concept_names[i]}. AUC set to 0.0")
    print("\nPer-concept AUC:")
    for name, auc in zip(concept_names, auc_scores):
        print(f"{name}: {auc:.3f}")
    print("\nPer-concept F1:")
    for i, name in enumerate(concept_names):
        f1 = f1_score(val_labels[:, i], y_pred_binary[:, i])
        print(f"{name}: {f1:.3f}")
    print("\nConfusion Matrices:")
    for i, name in enumerate(concept_names):
        cm = confusion_matrix(val_labels[:, i], y_pred_binary[:, i])
        print(f"{name}:\n{cm}")
except Exception as e:
    msg = f"Evaluation error: {e}"
    write_log(msg)

# --- Save ---
model.save_weights(FINAL_MODEL_PATH)
print("Final model saved")
if tb_ds is not None:
    try:
        concept_preds = model.predict(tb_ds, batch_size=4)
        np.save(PREDICTIONS_PATH, concept_preds)
        print("Concept predictions saved")
    except Exception as e:
        msg = f"Error saving concept predictions: {e}"
        write_log(msg)
else:
    print("Skipping concept predictions.")