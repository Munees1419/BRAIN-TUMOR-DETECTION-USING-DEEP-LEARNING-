import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
#streamlit run main1.py
#python -m venv venv     venv\Scripts\activate


# ---------------- Load Models ----------------
@st.cache_resource
def load_models():
    classification_model = load_model(r"C:\munees\MRI1407_VGG_model.h5")
    segmentation_model = load_model(r"C:\munees\tumor_segmentation_unet.h5")
    return classification_model, segmentation_model


# ---------------- Preprocess Image ----------------
def load_and_preprocess_image(img_path):
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    x = preprocess_input(resized.astype(np.float32))
    x = np.expand_dims(x, axis=0)
    return rgb, x

# ---------------- Prediction ----------------
def predict_tumor(image_tensor):
    prob = classification_model.predict(image_tensor)[0][0]
    return prob > 0.5, prob

# ---------------- Segmentation ----------------
def segment_tumor(original_rgb):
    SEG_IMG_SIZE = (128, 128)
    resized = cv2.resize(original_rgb, SEG_IMG_SIZE)
    x = resized.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    pred_mask = segmentation_model.predict(x)[0, :, :, 0]
    mask = (pred_mask > 0.1).astype(np.uint8)
    mask = cv2.resize(mask, (original_rgb.shape[1], original_rgb.shape[0]))
    return mask

# ---------------- Streamlit UI ----------------
st.title("🧠 Brain Tumor Classification & Segmentation")

folder_path = st.text_input("Enter folder path with images:")

segment_flag = st.checkbox("Segment tumor if detected", value=False)
show_each_flag = st.checkbox("Show results for each image", value=True)

classification_model, segmentation_model = load_models()
IMG_SIZE = (224, 224)

if st.button("Analyze Folder") and folder_path:
    if not os.path.exists(folder_path):
        st.error("Folder not found!")
    else:
        image_paths = []
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            image_paths.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(ext.split('*')[-1])])

        results = []

        for img_path in image_paths:
            try:
                rgb, x = load_and_preprocess_image(img_path)
                is_tumor, prob = predict_tumor(x)

                if show_each_flag:
                    st.write(f"{'Tumor' if is_tumor else 'No Tumor'}: {os.path.basename(img_path)} (p={prob:.2f})")

                overlay_image = None
                if is_tumor and segment_flag:
                    mask = segment_tumor(rgb)
                    overlay_image = rgb.copy()
                    overlay_image[mask == 1] = [255, 0, 0]

                results.append({
                    "filename": os.path.basename(img_path),
                    "tumor": int(is_tumor),
                    "no_tumor": int(not is_tumor),
                    "probability": float(prob),
                    "overlay": overlay_image
                })
            except Exception as e:
                st.warning(f"Error processing {img_path}: {e}")

        df = pd.DataFrame(results)
        st.success(" Analysis Complete")
        st.dataframe(df[['filename', 'tumor', 'probability']])

        # Show overlay images
        if segment_flag:
            st.subheader("Segmentation Overlays")
            for res in results:
                if res["overlay"] is not None:
                    st.image(res["overlay"], caption=res["filename"])

        # Plot summary
        st.subheader("Summary Plots")
        tumor_counts = df['tumor'].value_counts()
        labels = ['No Tumor', 'Tumor']
        sizes = [tumor_counts.get(0, 0), tumor_counts.get(1, 0)]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=[f"{label} ({count})" for label, count in zip(labels, sizes)],
                autopct='%1.1f%%', colors=["green", "red"])
        ax1.set_title("Tumor Classification Summary")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.hist(df[df["tumor"] == 1]["probability"], bins=10, alpha=0.7, label="Tumor", color='red')
        ax2.hist(df[df["tumor"] == 0]["probability"], bins=10, alpha=0.7, label="No Tumor", color='green')
        ax2.set_xlabel("Prediction Probability")
        ax2.set_ylabel("Number of Images")
        ax2.set_title("Tumor Prediction Probability Distribution")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        df.to_csv("tumor_classification_results.csv", index=False)
        st.success("Results saved to tumor_classification_results.csv")
