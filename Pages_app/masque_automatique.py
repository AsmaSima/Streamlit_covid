import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from utils.preprocess import masking
from utils.models import load_mask_model

def run():
    st.title("Masque automatique")
    st.write("""
            Afin d'améliorer l'entraînement du modèle Inception V3 et de réduire l'influence des artefacts et marquages médicaux
             présents sur les radiographies, un masque automatique a été appliqué aux images.
             Ce masque permet de ne concerver que la zone d'intérêt, à savoir les poumons.
             Pour cela, un modèle de segmentation basé sur PyTorch a été développé, entraîné et évalué.
             """)

    onglet1, onglet2, onglet3 = st.tabs(["Pré-traitement", "Entraînement", "Evaluation"])
    with onglet1:
        st.header("Pré-traitement")
        st.write("""
                L'ensemble des données disponibles a été utilisé pour l'entraînement du modèle de masque automatique. 
                Les données ont été réparties en trois ensembles distincts:
                - Ensemble d'entraînement : 70%
                - Ensemble de validation : 15%
                - Ensemble de test : 15%
                 """)
        
        df = pd.DataFrame({"Ensemble":["Train","Validation","Test"],
                           "Images":[14823,3167,3175]})
        colors={"Train":"#D070B5",
                "Validation":"#9A5DAC",
                "Test":"#68AECE"}
        fig = px.pie(df, names="Ensemble",
                     values="Images", hole=0.35,
                     color="Ensemble",
                     color_discrete_map=colors)
        fig.update_traces(textposition="inside",
                          textinfo="label+percent",
                          texttemplate="%{label} %{percent:.0%}",
                          insidetextorientation="radial")
        fig.update_layout(paper_bgcolor="#E8F0F8",
                          plot_bgcolor="#E8F0F8",
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        with onglet2:
            st.header("Entraînement")
            st.write("""
                    Pour concevoir le modèle de segmentation, nous avons retenu PyTorch comme framework principal, mieux adapté à la problématique abordée.
                    L'architecture repose sur un encodeur Resnet18 pré-entraîné sur ImageNet, afin de bénéficier d'un apprentissage par transfert.
                     Les images ont été redimensionnées à une taille 192x192 pixels.
                    Ces choix ont été effectués à l'issue de plusieurs tests et correspondent au meilleur compromis entre performance et temps de calcul.
                    Les principaux paramètres d'entraînement sont les suivants:
                    - Optimiseur : Adam
                    - Learning rate : 1.10<sup>-3</sup>
                    - Epochs : 5
                    """, unsafe_allow_html=True)
            
            df_mask = pd.DataFrame({"train_loss":[0.0432,0.0254,0.0232,0.0216,0.0202],
                                    "train_dice":[0.9680,0.9788,0.9806,0.9818,0.9829],
                                    "val_loss":[0.0349,0.0229,0.0285,0.0210,0.0201],
                                    "val_dice":[0.9712,0.9806,0.9757,0.9821,0.9829]})
            
            df_mask["epoch"]=range(1,len(df_mask)+1)

            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=df_mask["epoch"],
                                            y=df_mask["train_loss"],
                                            mode="lines+markers",
                                            name="Train loss",
                                            line=dict(color="#C2185B", dash="dash")))
            fig_loss.add_trace(go.Scatter(x=df_mask["epoch"],
                                            y=df_mask["val_loss"],
                                            mode="lines+markers",
                                            name="Validation loss",
                                            line=dict(color="#2E75C1")))
            fig_loss.update_layout(title="Evolution de la loss",
                                    xaxis_title="Epochs",
                                    yaxis_title="Loss",
                                    paper_bgcolor="#E8F0F8",
                                    plot_bgcolor="#E8F0F8",
                                    margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_loss, use_container_width=True)
            
        
            fig_dice = go.Figure()
            fig_dice.add_trace(go.Scatter(x=df_mask["epoch"],
                                            y=df_mask["train_dice"],
                                            mode="lines+markers",
                                            name="Train Dice",
                                            line=dict(color="#C2185B", dash="dash")))
            fig_dice.add_trace(go.Scatter(x=df_mask["epoch"],
                                            y=df_mask["val_dice"],
                                            mode="lines+markers",
                                            name="Validation Dice",
                                            line=dict(color="#2E75C1")))
            fig_dice.update_layout(title="Evolution du Dice score",
                                    xaxis_title="Epochs",
                                    yaxis_title="Dice",
                                    paper_bgcolor="#E8F0F8",
                                    plot_bgcolor="#E8F0F8",
                                    margin=dict(l=0,r=0,t=40,b=0),
                                    yaxis=dict(range=(0.95,1.0)))
            st.plotly_chart(fig_dice, use_container_width=True)

            st.write("""
                    **Interprétation :**
                    - La diminution progressive de la loss indique une convergence stable du modèle.
                    - Les scores Dice élevés et proches entre entraînement et validation suggèrent une bonne généralisation,
                     sans sur-apprentissage notable sur les données de validation.
                     """)
            
        with onglet3:
            st.header("Evaluation")
            st.write("""
                    Le meilleur modèle retenu après la phase d'entraînement est évalué sur un jeu de test indépendant
                    et atteint une performance globale de 98,3%.
                     
                    L'exemple ci-dessous illustre l'application du masque de segmentation des poumons sur une radiographie thoracique
                     aléatoire. L'utilisateur peut générer une nouvelle radiographie et activer ou désactiver le masque afin de comparer
                     visuellement les résultats.
                     """)
            
            mask_model = load_mask_model()

            def random_image(img_dir):
                img_name = random.choice(os.listdir(img_dir))
                img_path = os.path.join(img_dir, img_name)
                img_gray = np.array(Image.open(os.path.join(img_path)).convert("L"), dtype=np.uint8)
                img_rgb = np.stack([img_gray, img_gray, img_gray], axis=-1)
                return img_rgb, img_name
            
            img_dir = "Datas"

            cols = st.columns([1,3,1])
            with cols[1]:
                if "img" not in st.session_state:
                    st.session_state.img, st.session_state.img_name = random_image(img_dir)
                    st.session_state.img_masked = None
                    st.session_state.mask_3c = None
            
                if st.button("Nouvelle radiographie"):
                    st.session_state.img, st.session_state.img_name = random_image(img_dir)
                    st.session_state.img_masked = None
                    st.session_state.mask_3c = None

                st.image(st.session_state.img, caption=f"Radiographie : {st.session_state.img_name}", width=400)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.toggle("Générer le masque"):
                    st.session_state.img_masked, st.session_state.mask_3c = masking(st.session_state.img, mask_model=mask_model)
                    st.image(st.session_state.mask_3c, caption="Masque", use_container_width=True)
                    with col2:
                        if st.session_state.mask_3c is not None:
                            apply_mask = st.toggle("Appliquer le masque", value=True)
                            if apply_mask:
                                st.image(st.session_state.img_masked/255.0,
                                    caption="Image masquée",
                                    use_container_width=True)
                