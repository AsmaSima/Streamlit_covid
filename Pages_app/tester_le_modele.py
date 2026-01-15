import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import io
import plotly.express as px
import matplotlib.pyplot as plt



import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from utils.models import load_inception_model, load_mask_model
from utils.preprocess import preprocess_inceptionv3, make_gradcam_heatmap, overlay_gradcam

inception_model = load_inception_model()
mask_model = load_mask_model()

def run():
    st.title("Tester le modèle")
    
    demo_dir = ["Datas"]
    choice_mode = st.radio("Source de l'image :", ["Charger une radiographie", "Choisir une radiographie"],
                           horizontal=True)
    uploaded_file = None
    if choice_mode == "Charger une radiographie":
        uploaded_file = st.file_uploader("Choisir une radiographie", type=["jpg", "jpeg", "png"])
    else:
        demo_files = []
        for d in demo_dir:
            if os.path.exists(d):
                demo_files += [os.path.join(d,f) for f in os.listdir(d)if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        demo_files = sorted(demo_files)
        if len(demo_files)==0:
            st.warning("Aucune radiographie trouvée dans Datas")
        else:
            demo_choice = st.selectbox("Choisir une radiographie", demo_files, format_func=lambda p: os.path.basename(p))
            img = Image.open(demo_choice).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            buf.name = os.path.basename(demo_choice)
            uploaded_file = buf
   
    if uploaded_file is not None:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        st.image(uploaded_file, caption="Image chargée", use_container_width=300)

    if st.button("Prédire"):
        if uploaded_file is not None:
            progress = st.progress(0)
            progress.progress(10)
            x, img_masked, filename, true_class = preprocess_inceptionv3(uploaded_file, mask_model)
            progress.progress(30)
     
            progress.progress(75)
            preds = inception_model.predict(x)
            progress.progress(100)
            pred_class = np.argmax(preds)
            
            prob_predite = preds[0][pred_class]    
            prob = round(float(prob_predite)*100, 2)

            classes = ["Autres", "Covid-19", "Sain"]

            st.subheader("Résultat de la prédiction")
            st.write("**Classe réelle :**", true_class)
            st.write(f"**Classe prédite :** {classes[pred_class]}")
            st.write("**Probabilité :**", prob, "%")

            probs = (preds[0]*100).round(2)
            df_probs = pd.DataFrame({"Classe" : classes,
                                    "Probabilité (%)" : probs})
            
            fig_pie = px.pie(df_probs,
                            names = "Classe",
                            values = "Probabilité (%)",
                            hole = 0.4)
            
            fig_pie.update_traces(textposition = "outside",
                            textinfo = "percent+label",
                            textfont_size = 14)
            
            fig_pie.update_layout(paper_bgcolor = "rgba(0,0,0,0)",
                            plot_bgcolor = "rgba(0,0,0,0)",
                            font = dict(color="#0B1A33"),
                            legend = dict(font=dict(size=18)),
                            height= 400)
            
            st.plotly_chart(fig_pie, use_container_width=True)
            st.dataframe(df_probs, use_container_width=True)

            st.subheader("Grad-CAM : Interprétabilité de la prise de décision")

            last_conv_name = "mixed10"
            heatmap = make_gradcam_heatmap(x, inception_model, last_conv_name, pred_index=pred_class)

            overlay = overlay_gradcam(heatmap, img_masked)

            fig, ax = plt.subplots(figsize=(5,5))
            im = ax.imshow(overlay)
            ax.axis("off")
            ax.set_title("Carte Grad-CAM du modèle Inception V3", fontsize=10, color="#0B1A33")

            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            norm = plt.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
            sm.set_array([])
            
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            ticks = np.linspace(0,1,6)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])
            cbar.set_label("Contribution à la décision (%)", rotation=270, labelpad=15, fontsize=9)
            cbar.ax.tick_params(colors="#0B1A33", labelsize=9)
            cbar.ax.yaxis.label.set_color("#0B1A33")
            st.pyplot(fig, transparent=True)
                
        else:
            st.error("Veuillez importer une image avant de procéder à la prédiction.")
            st.stop()


                    