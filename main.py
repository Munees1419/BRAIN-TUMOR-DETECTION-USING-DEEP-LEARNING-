import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# Load models
classification_model = load_model(r"C:\munees\MRI1407_VGG_model.h5")
segmentation_model = load_model(r"C:\munees\tumor_segmentation_unet.h5")

IMG_SIZE = (224, 224)

# ---------------- Classification Preprocess (VGG16) ----------------
def load_and_preprocess_image(img_path):
    """
    Reads and preprocesses image for VGG16 classification model.
    """
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)

    x = preprocess_input(resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)

    return rgb, x

# ---------------- Classification ----------------
def predict_tumor(image_tensor):
    """
    Predict tumor presence using classification model.
    """
    prob = classification_model.predict(image_tensor)[0][0]
    return prob > 0.5, prob


def segment_tumor(original_rgb):
    """
    Segment tumor using U-Net model (expects 3-channel input).
    """
    SEG_IMG_SIZE = (128, 128)

    # Keep as RGB (3-channel) since your model was trained with RGB input
    resized = cv2.resize(original_rgb, SEG_IMG_SIZE)   # shape (128,128,3)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)   # shape (1,128,128,3)

    pred_mask = segmentation_model.predict(x)[0, :, :, 0]  # shape (128,128)

    # Debug: check prediction stats
    print("Mask stats -> min:", pred_mask.min(),
          "max:", pred_mask.max(),
          "mean:", pred_mask.mean())

    # Threshold
    #mask= ( pred_mask> 0.5).astype(np.uint8)
    mask= (pred_mask > 0.1).astype(np.uint8)
    #print("Mask range:", mask.min(), mask.max(), mask.mean())

    #plt.imshow(mask, cmap='gray')



    # Resize back to original image size
    mask = cv2.resize(mask, (original_rgb.shape[1], original_rgb.shape[0]))

    return mask

# ---------------- Display Results ----------------
def show_results(rgb, mask=None, prob=None):
    """
    Display results with matplotlib.
        """
    os.makedirs('output/plots', exist_ok=True)

    if mask is not None:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(rgb)
        plt.title("Original")
        plt.savefig("output/plots/tumor2.png")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.savefig("output/plots/tumor1.png")
        plt.imshow(mask, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        overlay = rgb.copy()
        overlay[mask == 1] = [255, 0, 0]   # red overlay
        plt.imshow(overlay)
        plt.title(f"masks (p={prob:.2f})")
        plt.axis("off")
        plt.savefig("output/plots/tumor.png")
        plt.show()
    else:
        plt.savefig("output/plots/no_tumor.png")
        plt.imshow(rgb)
        plt.title(f"No Tumor Detected (p={prob:.2f})")
        plt.savefig("output/plots/no_tumor1.png")
        plt.axis("off")
        plt.show()

# ---------------- Full Pipeline ----------------
def analyze_image(img_path):
    """
    Full pipeline: classification + segmentation + visualization.
    """
    try:
        rgb, x = load_and_preprocess_image(img_path)
        is_tumor, prob = predict_tumor(x)

        if is_tumor:
            print("Tumor Detected")
            mask = segment_tumor(rgb)
            plt.savefig("output/plots/tumor.png")
            show_results(rgb, mask, prob)
            
        else:
            print(" No Tumor Detected")
            
            show_results(rgb, None, prob)
            

    except Exception as e:
        print(f"⚠️ Error: {e}")



# ---------------- Run ----------------
analyze_image(r"C:\munees\testing images\no_tumor\noo_004.png")
#analyze_image(r"C:\munees\testing images\no_tumor\noo_117.jpg")
#analyze_image(r"C:\munees\testing images\tumor\yes_119.png")
analyze_image(r"C:\munees\testing images\tumor\yes_124.png")

