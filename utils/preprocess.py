import streamlit as st
import numpy as np
import cv2

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
import torch

def masking(img_rgb, mask_model, device="cpu", mask_size=192):
    """
    img_rgb : uint8 ou float32, shape(H,W,3), valeurs normales
    retourne : img masquée, même shape
    """
    img_small = cv2.resize(img_rgb, (mask_size, mask_size))
    x = img_small.astype(np.float32)/255.0
    x = np.transpose(x,(2,0,1))
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        mask = torch.sigmoid(mask_model(x))[0,0].cpu().numpy()
    mask = (mask>0.5).astype(np.float32)
    mask = cv2.resize(mask,(img_rgb.shape[1],img_rgb.shape[0]))
    mask_3c = np.repeat(mask[...,None],3,axis=2)
    return img_rgb*mask_3c, mask_3c

def preprocess_inceptionv3(uploaded_file, mask_model, device="cpu"):
    """
    Pipeline pour InceptionV3:
    -Lecture de l'image uploadée
    -Applique CLAHE
    -Resize de l'image en 256x256
    -Chercher et applique le mask correspondant
    -Normalise avec preprocess_input
    Retourne : 
    x : np.array pour le modèle
    img_rgb_299 : image RGB en 299x299 masquée (pour affichage streamlit)
    filename : nom du ficher
    """
    uploaded_file.seek(0)
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Impossible de décoder l'image uploadée.")
    filename = uploaded_file.name.lower()
    if "covid" in filename:
        true_class = "Covid-19"
    elif "sain" in filename or "normal" in filename:
        true_class = "Sain"
    elif "opacity" in filename or "pneumonia" in filename:
        true_class = "Autres"
    else:
        true_class = "Inconnue"

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe,a,b))
    img_bgr_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    img_bgr_256 = cv2.resize(img_bgr_clahe, (256,256))

    img_rgb_256 = cv2.cvtColor(img_bgr_256, cv2.COLOR_BGR2RGB)
    
    img_rgb_masked_256,_ = masking(img_rgb_256, mask_model, device, mask_size = 192)
    img_rgb_masked_299 = cv2.resize(img_rgb_masked_256, (299,299))

    x = img_rgb_masked_299.astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    return x, img_rgb_masked_299, filename, true_class

def make_gradcam_heatmap(img, model, last_conv_layer_name, pred_index=None):
    """
    img : np.array shape (1,H,W,3) preprocessé
    pred_index : indice de la classe prédite
    """
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    if len(img.shape) == 3:
        img = tf.expand_dims(img, axis=0)

    last_conv_layer = model.get_layer(last_conv_layer_name)    
    grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img, training=False)
        if isinstance(preds, (list,tuple)):
            preds = preds[-1]
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap,0)/(tf.reduce_max(heatmap)+1e-8)
    return heatmap.numpy()

def overlay_gradcam(heatmap, img, alpha=0.4):
    """
    heatmap : 2D(H,W)
    img : image RGB uint8 (H,W,3)
    """
    h,w = img.shape[:2]
    heatmap_resize = cv2.resize(heatmap, (w,h))
    heatmap_uint8 = np.uint8(255*heatmap_resize)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    superimposed = np.uint8(alpha*heatmap_color + (1- alpha)*img)
    return superimposed