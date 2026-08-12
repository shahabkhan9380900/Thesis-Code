import sys
import os
import traceback

if getattr(sys, 'frozen', False):
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

import cv2
import pywt
import numpy as np
from skimage.feature import hog
from keras.models import load_model
import joblib
from keras.models import Model
import tensorflow as tf
import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
from PIL import Image

def safe_messagebox(title, message):
    try:
        messagebox.showerror(title, message)
    except Exception:
        pass


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


try:
    cnn = load_model(resource_path("../models/cnn_for_new_identification.h5"), compile=False)

    rf = joblib.load(resource_path("../models/rf_new_model.pkl"))
    scal_rf = joblib.load(resource_path("../models/scaler_rf_new_model.pkl"))
    PCA_rf = joblib.load(resource_path("../models/PCA_RF_new_model.pkl"))

    svm = joblib.load(resource_path("../models/svm_new_model.pkl"))
    scal_svm = joblib.load(resource_path("../models/scaler_svm .pkl"))
    PCA_svm = joblib.load(resource_path("../models/PCA_SVM_new_model.pkl"))

    knn = joblib.load(resource_path("../models/knn_new_model.pkl"))
    scal_knn = joblib.load(resource_path("../models/scaler_knn_new_model.pkl"))
    PCA_knn = joblib.load(resource_path("../models/PCA_KNN_new_model.pkl"))

    LogR = joblib.load(resource_path("../models/logr_new_model.pkl"))
    scal_lr = joblib.load(resource_path("../models/scaler_LR_new_model.pkl"))
    PCA_lr = joblib.load(resource_path("../models/PCA_LR_new_model.pkl"))

except Exception as e:
    print("❌ Model Loading Error:", e)
    traceback.print_exc()
    safe_messagebox("Model Loading Error", str(e))

# ------------------------------
# Global Variables
# ------------------------------
gallery_items = []
popup_windows = []  
MAX_COLUMNS = 5
CARD_PADDING = 1
SIDE_MARGIN = 10
SCROLL_RIGHT_PADDING = 5
cnn_feature_model = Model(inputs=cnn.input, outputs=cnn.layers[-3].output)  


def extract_cnn_features_batch(images):
    images = images.astype(np.float32) / 255.0
    if images.ndim == 3:
        images = images[..., np.newaxis]
    return cnn_feature_model(images, training=False).numpy().reshape(len(images), -1)

def extract_wavelet_features(images, wavelet="db2", level=2):
    feats = []
    for img in images:
        coeffs = pywt.wavedec2(img.astype(np.float32), wavelet=wavelet, level=level)
        cA = coeffs[0]
        cH, cV, cD = coeffs[1]
        feats.append(np.hstack([np.nan_to_num(cA).ravel(), np.nan_to_num(cH).ravel()]))
    return np.array(feats, dtype=np.float32)

def extract_hog_features(images):
    feats = []
    for img in images:
        feat = hog(img.astype(np.float32)/255.0, orientations=8, pixels_per_cell=(16,16),
                   cells_per_block=(1,1), block_norm="L2-Hys", visualize=False, feature_vector=True)
        feats.append(np.nan_to_num(feat))
    return np.array(feats, dtype=np.float32)

def extract_hybrid_features(images):
    return np.hstack([extract_wavelet_features(images),
                      extract_hog_features(images),
                      extract_cnn_features_batch(images)]).astype(np.float32)


def calculate_thumbnail_size():
    available_width = screen_width - 2*SIDE_MARGIN - SCROLL_RIGHT_PADDING - (MAX_COLUMNS-1)*CARD_PADDING
    thumb_size = available_width // (MAX_COLUMNS + 1)
    return max(290, thumb_size)

def open_image_popup(card):
    popup = ctk.CTkToplevel(root)
    popup.title("Image Options")
    popup.geometry("500x500")
    popup.transient(root)
    popup.lift()
    popup.protocol("WM_DELETE_WINDOW", popup.destroy)
    popup_windows.append(popup)

    thumb_size = min(250, card.img.width, card.img.height)
    img = card.img.resize((thumb_size, thumb_size))
    img_tk = CTkImage(light_image=img, size=(thumb_size, thumb_size))
    img_label = ctk.CTkLabel(popup, image=img_tk, text="")
    img_label.image = img_tk
    img_label.pack(pady=20)

    def zoom_image():
        zoom_popup = ctk.CTkToplevel(root)
        zoom_popup.title("Zoom Image")
        zoom_popup.transient(root)
        zoom_popup.lift()
        zoom_popup.protocol("WM_DELETE_WINDOW", zoom_popup.destroy)
        popup_windows.append(zoom_popup)

        original = card.img
        w, h = original.size
        double_w = min(w*2, screen_width-50)
        double_h = min(h*2, screen_height-50)

        img_zoom = original.resize((double_w, double_h))
        img_zoom_tk = CTkImage(light_image=img_zoom, size=(double_w, double_h))
        label_zoom = ctk.CTkLabel(zoom_popup, image=img_zoom_tk, text="")
        label_zoom.image = img_zoom_tk
        label_zoom.pack()

    def remove_image():
        if card in gallery_items:
            card.destroy()
            gallery_items.remove(card)
            arrange_gallery()
        popup.destroy()

    btn_frame = ctk.CTkFrame(popup, fg_color="transparent")
    btn_frame.pack(pady=10)
    ctk.CTkButton(btn_frame, text="🔍 Zoom", command=zoom_image, width=100).pack(side="left", padx=10)
    ctk.CTkButton(btn_frame, text="🗑 Remove", command=remove_image, width=100).pack(side="left", padx=10)

def add_to_gallery(file_path):
    try:
        thumb_size = calculate_thumbnail_size()
        card = ctk.CTkFrame(inner_gallery_frame, corner_radius=15, fg_color="#252525")

        img = Image.open(file_path).convert("L").resize((200,200))
        card.img_small = np.array(img)/255.0
        card.img = Image.open(file_path)
        card.file_path = file_path

        # Precompute features for fast prediction
        card.wave_feat = extract_wavelet_features([card.img_small])[0]
        card.hog_feat = extract_hog_features([card.img_small])[0]

        img_thumb = img.resize((thumb_size, thumb_size))
        img_tk = CTkImage(light_image=img_thumb, size=(thumb_size, thumb_size))

        img_container = ctk.CTkFrame(card, width=thumb_size, height=thumb_size, corner_radius=15, fg_color="#1a1a1a")
        img_container.pack_propagate(False)
        img_container.pack(padx=5, pady=5)
        img_label = ctk.CTkLabel(img_container, image=img_tk, text="")
        img_label.image = img_tk
        img_label.pack(expand=True, fill="both")

        file_name = os.path.basename(file_path)
        name_label = ctk.CTkLabel(card, text=file_name, text_color="black", font=("Arial", 12))
        name_label.pack(pady=(0,5))

        card.bind("<Button-1>", lambda e, c=card: open_image_popup(c))
        img_container.bind("<Button-1>", lambda e, c=card: open_image_popup(c))
        img_label.bind("<Button-1>", lambda e, c=card: open_image_popup(c))

        gallery_items.append(card)
        arrange_gallery()
    except Exception as e:
        print("❌ add_to_gallery error:", e)
        traceback.print_exc()
        safe_messagebox("Add to Gallery Error", str(e))

def arrange_gallery():
    try:
        # Remove old widgets
        for widget in gallery_frame.winfo_children():
            widget.grid_forget()
        
        # Arrange new widgets
        for index, card in enumerate(gallery_items):
            row = index // MAX_COLUMNS
            col = index % MAX_COLUMNS
            card.grid(row=row, column=col, padx=CARD_PADDING, pady=CARD_PADDING)
        
        # Force layout update
        gallery_frame.update_idletasks()

       

    except Exception as e:
        print("❌ arrange_gallery error:", e)


def select_images():
    try:
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
        for file_path in file_paths:
            add_to_gallery(file_path)
        pred_text.set("Prediction: None")
    except Exception as e:
        print("❌ select_images error:", e)
        traceback.print_exc()
        safe_messagebox("Select Images Error", str(e))

def remove_all_images():
    global inner_gallery_frame
    try:
        # Remove all images from the gallery
        for widget in inner_gallery_frame.winfo_children():
            widget.destroy()
        gallery_items.clear()
        pred_text.set("Prediction: None")
        summary_label_top.configure(text="")
        summary_label_bottom.configure(text="")

        # Destroy old inner frame and create a new one
        inner_gallery_frame.destroy()
        inner_gallery_frame = ctk.CTkFrame(gallery_frame, fg_color="transparent")
        inner_gallery_frame.pack(fill="both", expand=True, padx=(0, SCROLL_RIGHT_PADDING))

        # Scroll gallery_frame to top
        gallery_frame._parent_canvas.yview_moveto(0)

    except Exception as e:
        print("❌ remove_all_images error:", e)
        traceback.print_exc()
        safe_messagebox("Remove All Images Error", str(e))


def predict_images():
    try:
        if not gallery_items:
            pred_text.set("Prediction: No images")
            summary_label_top.configure(text="")
            summary_label_bottom.configure(text="")
            return

        # Use precomputed wavelet & HOG, batch CNN
        imgs_small = np.array([card.img_small for card in gallery_items])
        cnn_feats = extract_cnn_features_batch(imgs_small)
        wave_feats = np.array([card.wave_feat for card in gallery_items])
        hog_feats = np.array([card.hog_feat for card in gallery_items])
        hybrid_feats = np.hstack([wave_feats, hog_feats, cnn_feats])

        # Ensemble Predictions
        preds_rf = rf.predict_proba(scal_rf.transform(PCA_rf.transform(hybrid_feats)))[:,1]
        preds_svm = svm.predict_proba(scal_svm.transform(PCA_svm.transform(hybrid_feats)))[:,1]
        preds_knn = knn.predict_proba(scal_knn.transform(PCA_knn.transform(hybrid_feats)))[:,1]
        preds_lr = LogR.predict_proba(scal_lr.transform(PCA_lr.transform(hybrid_feats)))[:,1]
        preds_cnn = cnn.predict(imgs_small.reshape(-1,200,200,1), verbose=0).flatten()

        preds_stack = np.vstack([preds_cnn, preds_rf, preds_svm, preds_knn, preds_lr]).T
        weights1=[0.20036765, 0.20289522 ,0.19692096 ,0.20519301 ,0.19462316]
        ensemble = (np.dot(preds_stack, weights1) >= 0.5).astype(int)

        for card, pred in zip(gallery_items, ensemble):
            if hasattr(card, "pred_label"):
                card.pred_label.destroy()
            text = "Tumor 🧠" if pred == 1 else "Healthy ✅"
            color = "#bd8080" if pred == 1 else "#56e4b0"
            card.configure(fg_color=color)
            card.pred_label = ctk.CTkLabel(card, text=text, text_color='black', font=("Arial", 14, "bold"))
            card.pred_label.pack(pady=5)

        total = len(ensemble)
        total_tumor = np.sum(ensemble == 1)
        total_healthy = np.sum(ensemble == 0)
        summary_text = f"📊 Prediction Summary 📊\n\nTotal Images: {total}\nTumor 🧠: {total_tumor}\nHealthy ✅: {total_healthy}"
        summary_label_top.configure(text=summary_text)
        summary_label_bottom.configure(text=summary_text)
        pred_text.set("Prediction: Done ✅")

    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        safe_messagebox("Prediction Error", str(e))
        pred_text.set("Prediction Failed")

def auto_predict_folder():
    try:
        
        img_path = resource_path("../data/unseen")  
        classes_in_dir = {"(test)Tumor": 1, "(test)Healty": 0}  
        remove_all_images()

        all_imgs, real_labels = [], []

        for cls_name, label in classes_in_dir.items():
            folder_path = os.path.join(img_path, cls_name)
            if not os.path.exists(folder_path):
                print(f"⚠️ Folder not found: {folder_path}")
                continue

            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    file_path = os.path.join(folder_path, file_name)
                    try:
                        img = Image.open(file_path).convert("L").resize((200, 200))
                        add_to_gallery(file_path)  # Show in GUI
                        all_imgs.append(np.array(img)/255.0)
                        real_labels.append(label)
                    except Exception as e:
                        print(f"❌ Error loading {file_path}: {e}")

        if not all_imgs:
            pred_text.set("No images found in folder")
            return

        all_imgs = np.array(all_imgs)
        cnn_feats = extract_cnn_features_batch(all_imgs)
        wave_feats = extract_wavelet_features(all_imgs)
        hog_feats = extract_hog_features(all_imgs)
        hybrid_feats = np.hstack([wave_feats, hog_feats, cnn_feats])
        preds_rf = rf.predict_proba(scal_rf.transform(PCA_rf.transform(hybrid_feats)))[:, 1]
        preds_svm = svm.predict_proba(scal_svm.transform(PCA_svm.transform(hybrid_feats)))[:, 1]
        preds_knn = knn.predict_proba(scal_knn.transform(PCA_knn.transform(hybrid_feats)))[:, 1]
        preds_lr = LogR.predict_proba(scal_lr.transform(PCA_lr.transform(hybrid_feats)))[:, 1]
        preds_cnn = cnn.predict(all_imgs.reshape(-1, 200, 200, 1), verbose=0).flatten()

        preds_stack = np.vstack([preds_cnn, preds_rf, preds_svm, preds_knn, preds_lr]).T
        weights = [0.20036765, 0.20289522, 0.19692096, 0.20519301, 0.19462316]
        ensemble = (np.dot(preds_stack, weights) >= 0.5).astype(int)
        for card, pred in zip(gallery_items, ensemble):
            if hasattr(card, "pred_label"):
                card.pred_label.destroy()
            text = "Tumor 🧠" if pred == 1 else "Healthy ✅"
            color = "#bd8080" if pred == 1 else "#56e4b0"
            card.configure(fg_color=color)
            card.pred_label = ctk.CTkLabel(card, text=text, text_color='black', font=("Arial", 14, "bold"))
            card.pred_label.pack(pady=5)
        real_labels = np.array(real_labels)
        total_tumor_real = np.sum(real_labels == 1)
        total_healthy_real = np.sum(real_labels == 0)
        total_tumor_pred = np.sum(ensemble == 1)
        total_healthy_pred = np.sum(ensemble == 0)
        correct_tumor = np.sum((ensemble == 1) & (real_labels == 1))
        correct_healthy = np.sum((ensemble == 0) & (real_labels == 0))
        tumor_acc = (correct_tumor / total_tumor_real * 100) if total_tumor_real > 0 else 0
        healthy_acc = (correct_healthy / total_healthy_real * 100) if total_healthy_real > 0 else 0
        overall_acc = ((correct_tumor + correct_healthy) / len(real_labels) * 100) if len(real_labels) > 0 else 0

        summary_text = (
            f"📊 Auto Prediction on Test Data 📊\n\n"
            f"🧠 Tumor: Real={total_tumor_real}, Predicted={total_tumor_pred}, Accuracy={tumor_acc:.2f}%\n"
            f"✅ Healthy: Real={total_healthy_real}, Predicted={total_healthy_pred}, Accuracy={healthy_acc:.2f}%\n"
            f"🔹 Overall Accuracy: {overall_acc:.2f}%"
        )
        summary_label_top.configure(text=summary_text)
        summary_label_bottom.configure(text=summary_text)

        pred_text.set("Auto Prediction Done ✅")

    except Exception as e:
        print("❌ Auto Predict Folder Error:", e)
        traceback.print_exc()
        safe_messagebox("Auto Predict Error", str(e))
        pred_text.set("Auto Prediction Failed")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
root = ctk.CTk()
root.title("IDENTIFICATION OF BRAIN TUMOR")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

title_label = ctk.CTkLabel(root, text="🌌 IDENTIFICATION OF BRAIN TUMOR",
                           font=("Arial Rounded MT Bold",34,"bold"),
                           text_color="#00c6ff")
title_label.pack(pady=20)

button_frame = ctk.CTkFrame(root, corner_radius=20, fg_color="#202020")
button_frame.pack(pady=10)

button_width, button_height, button_font = 200, 50, ("Arial", 18, "bold")

ctk.CTkButton(button_frame, text="📂 Select Images", command=select_images,
              width=button_width, height=button_height, corner_radius=15,
              fg_color="#4da6ff", hover_color="#3399ff", text_color="#ffffff",
              font=button_font).pack(side="left", padx=15)

ctk.CTkButton(button_frame, text="🗑 Remove All", command=remove_all_images,
              width=button_width, height=button_height, corner_radius=15,
              fg_color="#ff6666", hover_color="#ff4d4d", text_color="#ffffff",
              font=button_font).pack(side="left", padx=15)

ctk.CTkButton(button_frame, text="🧠 Predict All", command=predict_images,
              width=button_width, height=button_height, corner_radius=15,
              fg_color="#56e4b0", hover_color="#34d1a5", text_color="#000000",
              font=button_font).pack(side="left", padx=15)

ctk.CTkButton(button_frame, text="📁 Auto Predict Folder", command=auto_predict_folder,
              width=button_width, height=button_height, corner_radius=15,
              fg_color="#f0a500", hover_color="#e59400", text_color="#000000",
              font=button_font).pack(side="left", padx=15)

pred_text = ctk.StringVar(value="Prediction: None")
ctk.CTkLabel(button_frame, textvariable=pred_text, font=("Arial", 18)).pack(side="left", padx=15)

summary_label_top = ctk.CTkLabel(root, text="", font=("Arial", 18), justify="center", text_color="#00ffea")
summary_label_top.pack(pady=5)

gallery_frame = ctk.CTkScrollableFrame(root, orientation="vertical", height=screen_height-250)
gallery_frame.pack(fill="both", expand=True, padx=SIDE_MARGIN, pady=10)

inner_gallery_frame = ctk.CTkFrame(gallery_frame, fg_color="transparent")
inner_gallery_frame.pack(fill="both", expand=True, padx=(0, SCROLL_RIGHT_PADDING))

summary_label_bottom = ctk.CTkLabel(root, text="", font=("Arial", 18), justify="center", text_color="#00ffea")
summary_label_bottom.pack(pady=5)

try:
    root.mainloop()
except Exception as e:
    print("❌ GUI Mainloop Error:", e)
    traceback.print_exc()
    safe_messagebox("GUI Error", str(e))

