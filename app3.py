import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ============================
# CONFIG
# ============================
IMG_SIZE_CLASS = (224,224)   # for classification
IMG_SIZE_SEG   = (128,128)   # for segmentation

DATASET_IMG_PATH = "C:\munees\mrimask\images"   # contains yes/no folders
DATASET_MASK_PATH = "C:\munees\mrimask\masks"   # contains masks only for yes

# ============================
# 1. CLASSIFICATION MODEL (Detection)
# ============================
train_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    DATASET_IMG_PATH,
    target_size=IMG_SIZE_CLASS,
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = train_gen.flow_from_directory(
    DATASET_IMG_PATH,
    target_size=IMG_SIZE_CLASS,
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMG_SIZE_CLASS+(3,))
base_model.trainable = False

clf_model = Sequential([
    base_model,
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

clf_model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

clf_model.fit(train_data, validation_data=val_data, epochs=15)
clf_model.save("tumor_detection_vgg16.h5")

# ============================
# 2. SEGMENTATION MODEL (U-Net)
# ============================
def simple_unet(img_size):
    inputs = layers.Input(img_size+(3,))
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D((2,2))(c2)

    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

    u1 = layers.UpSampling2D((2,2))(b)
    concat1 = layers.concatenate([u1, c2])
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(concat1)

    u2 = layers.UpSampling2D((2,2))(c3)
    concat2 = layers.concatenate([u2, c1])
    c4 = layers.Conv2D(32, 3, activation="relu", padding="same")(concat2)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c4)
    return Model(inputs, outputs)

# Load images + masks (only for YES class)
def load_images_and_masks(img_dir, mask_dir, img_size):
    X, Y = [], []
    for fname in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, fname))
        if img is None: continue
        img = cv2.resize(img, img_size)

        mask_path = os.path.join(mask_dir, fname.replace(".jpg", "_mask.png"))
        if not os.path.exists(mask_path):
            continue  # only yes class has masks
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)

        X.append(img/255.0)
        Y.append(mask[..., np.newaxis]/255.0)
    return np.array(X), np.array(Y)

X_yes, Y_yes = load_images_and_masks(os.path.join(DATASET_IMG_PATH,"yes"),
                                     os.path.join(DATASET_MASK_PATH,"yes"),
                                     IMG_SIZE_SEG)

X_train, X_val, y_train, y_val = train_test_split(X_yes, Y_yes, test_size=0.2, random_state=42)

seg_model = simple_unet(IMG_SIZE_SEG)
seg_model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
seg_model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=20, batch_size=8)

seg_model.save("tumor_segmentation_unet.h5")

# ============================
# 3. INFERENCE PIPELINE
# ============================
def predict_pipeline(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("⚠️ Image not found")
        return
    
    # ---- Detection ----
    img_clf = cv2.resize(img, IMG_SIZE_CLASS)/255.0
    pred = clf_model.predict(img_clf[np.newaxis,...])[0][0]

    if pred < 0.5:
        print("❌ No Tumor Detected")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image - No Tumor")
        plt.axis("off")
        plt.show()
        return
    
    print("✅ Tumor Detected")

    # ---- Segmentation ----
    img_seg = cv2.resize(img, IMG_SIZE_SEG)/255.0
    mask_pred = seg_model.predict(img_seg[np.newaxis,...])[0]

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(mask_pred.squeeze(), cmap="gray")
    plt.title("Predicted Tumor Mask")
    plt.axis("off")
    plt.show()

# Example
predict_pipeline("C:\munees\mrimask\images\yes\3.png")




"""this show red overlay
def predict_pipeline_with_overlay(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("⚠️ Image not found")
        return
    
    # ---- Detection ----
    img_clf = cv2.resize(img, IMG_SIZE_CLASS)/255.0
    pred = clf_model.predict(img_clf[np.newaxis,...])[0][0]

    if pred < 0.5:
        print("❌ No Tumor Detected")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image - No Tumor")
        plt.axis("off")
        plt.show()
        return
    
    print("✅ Tumor Detected")

    # ---- Segmentation ----
    img_seg = cv2.resize(img, IMG_SIZE_SEG)/255.0
    mask_pred = seg_model.predict(img_seg[np.newaxis,...])[0].squeeze()

    # Resize mask back to original image size
    mask_resized = cv2.resize((mask_pred*255).astype(np.uint8), (img.shape[1], img.shape[0]))

    # Create overlay (red mask)
    overlay = img.copy()
    overlay[mask_resized > 128] = [255, 0, 0]   # red color

    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Show results
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Tumor Overlay")
    plt.axis("off")

    plt.show()

# Example
predict_pipeline_with_overlay("test.jpg")
"""



"""3 this is check tumor or no tumor and save the image also

def predict_pipeline_with_overlay(img_path, save_path="output_with_mask.jpg"):
    img = cv2.imread(img_path)
    if img is None:
        print("⚠️ Image not found")
        return
    
    # ---- Detection ----
    img_clf = cv2.resize(img, IMG_SIZE_CLASS)/255.0
    pred = clf_model.predict(img_clf[np.newaxis,...])[0][0]

    if pred < 0.5:
        print("❌ No Tumor Detected")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image - No Tumor")
        plt.axis("off")
        plt.show()
        return
    
    print("✅ Tumor Detected")

    # ---- Segmentation ----
    img_seg = cv2.resize(img, IMG_SIZE_SEG)/255.0
    mask_pred = seg_model.predict(img_seg[np.newaxis,...])[0].squeeze()

    # Resize mask back to original image size
    mask_resized = cv2.resize((mask_pred*255).astype(np.uint8), (img.shape[1], img.shape[0]))

    # Create overlay (red mask)
    overlay = img.copy()
    overlay[mask_resized > 128] = [255, 0, 0]   # red color

    blended = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # ---- Save Output ----
    cv2.imwrite(save_path, blended)
    print(f"💾 Output saved at: {save_path}")

    # ---- Show Results ----
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(mask_resized, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    plt.title("Tumor Overlay")
    plt.axis("off")

    plt.show()

# Example usage
predict_pipeline_with_overlay("test.jpg", "tumor_overlay_output.jpg")
"""