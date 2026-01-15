import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import cv2


def run():
    st.title("Modélisation - Inception V3")
    st.write("""
            Cette partie est inspirée du travail réalisé par Alqatani _et al._
             [<sup>[2]</sup>](https://www.techscience.com/iasc/v35n2/48867/html).
            Ils utilisent le modèle InceptionV4 pour la détection du Covid-19 à partir de radiographies thoraciques,
            qui a conduit à d'excellents résultats.
            Nous avons, donc, adapté leur méthodologie à notre jeu de données sur le modèle InceptionV3
             [<sup>[3]</sup>](https://arxiv.org/abs/1512.00567).
            """, unsafe_allow_html=True)


    onglet1, onglet2, onglet3 = st.tabs(["Pré-traitement", "Entraînement", "Evaluation"])
    with onglet1:
        st.header("Pré-traitement")
        st.subheader("Sélection & séparation des données")
        st.write("""
                Les données sont réparties en trois classes :
                - Sain : 3500 images
                - Covid-19 : 3500 images
                - Autres (Opacité pulmonaire + Pneumonie virale) : 3500 images
                """)
        st.write("""
                Après traitement, les images ont été séparées en 3 ensembles :
                - Ensemble d'entraînement : 70%
                - Ensemble de validation : 15%
                - Ensemble de test : 15%
                """)
        cat = ["Normal", "Covid-19", "Autres"]
        total_par_classe = [3610, 3610, 3610]

        rows = []
        for c, total in zip(cat, total_par_classe):
            train = int(total*0.85)
            val = int(train*0.176)
            test = total-train
            rows.append({"Catégorie" : c, "Ensemble" : "Train", "Valeur": train})
            rows.append({"Catégorie" : c, "Ensemble" : "Validation", "Valeur": val})
            rows.append({"Catégorie" : c, "Ensemble" : "Test", "Valeur": test})
                
            df = pd.DataFrame(rows)

            fig = px.sunburst(df, path=["Catégorie","Ensemble"],
                                  values="Valeur",
                                  color="Catégorie",
                                  color_discrete_map={"Normal":"#D070B5",
                                                      "Covid-19":"#9A5DAC",
                                                      "Autres":"#68AECE"})
        fig.update_layout(paper_bgcolor="#E8F0F8",
                              plot_bgcolor="#E8F0F8",
                              margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)

        #--------Pipeline---------#
        from utils.models import load_mask_model
        from utils.preprocess import masking

        mask_model = load_mask_model()

        st.subheader("Pipeline de pré-traitement")
        st.write("""
                Les radiographies sont soumises à une chaîne de pré-traitement visant à améliorer la qualité
                 des images et à standardiser les entrés du modèle. 
                 
                 Les étapes sont les suivantes:
                - **Amélioration du contraste** à l'aide de la méthode _CLAHE_ afin de mettre en évidence
                  les structures pulmonaires et les détails pathologiques.
                - **Génération automatique du masque** à l'aide du modèle de segmentation.
                - **Application du masque** pour supprimer les zones non pertinentes de la radiographie.
                - **Redimensionnement en 299x299 pixels**, format requis par l'architecture Inception V3.
                - **Standardisation entre [-1,1]** _via_ la fonction _preprocess_input_ d'Inception V3.
                 """)

        img_gray = cv2.imread("Datas/COVID-32.png", cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            st.error("Image introuvable")

        col1,col2 = st.columns([2,1])
        with col1:
            clip_limit = st.slider("Clip Limit", 1.0, 5.0, 2.0, 1.0)
            tile_size = st.slider("tileGridSize", 4, 16, 8, 2)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size,tile_size))
            img_clahe = clahe.apply(img_gray)
        with col2:
            show_mask = st.checkbox("Ajouter le masque")

        col1,col2, col3 = st.columns(3)
        with col1:
            st.image(img_gray, caption="Image originale", use_container_width=True)
        with col2:
            st.image(img_clahe, caption=f"Après CLAHE (clipLimit={clip_limit}, tileGridSize=({tile_size}x{tile_size})",
                     channels="GRAY", use_container_width=True)
        with col3:
            if show_mask:
                img_clahe_rgb = np.stack([img_clahe]*3, axis=-1)
                img_mask, mask = masking(img_clahe_rgb, mask_model)
                st.image(img_mask/255.0, caption="Image avec masque appliqué", use_container_width=True)
            else:
                st.info("Cochez la case pour appliquer le masque")
        
        st.write("""
                Ce pipeline permet de focaliser l'apprentissage du modèle sur les régions pulmonaires tout en garantissant
                 des entrées homogènes et compatibles avec le réseau de neurones.
                 """)

    with onglet2:
        st.header("Entraînement")
        st.write("""
                L'entraînement du modèle s'es déroulé en plusieurs phases. Une première étape a consisté à entraîner 
                 uniquement la tête de classification afin d'adapter le modèle pré-entraîné à la tâche ciblée, à savoir
                 une classification en trois classes. Durant cette phase, les couches profondes du réseau ont été
                 conservées gelées afin de préserver les représentations apprises initialement.
                 """, unsafe_allow_html=True)
        with open("Images/history_inceptionv3_baseline.json", "r") as f:
            history = json.load(f)
        df = pd.DataFrame(history)
        df["epoch"] = range(1, len(df)+1)

        cols=st.columns(2)
        with cols[0]:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["loss"],
                                      mode="lines+markers", name="Train loss"))
        
            fig_loss.add_trace(go.Scatter(x=df["epoch"], y=df["val_loss"],
                                      mode="lines+markers", name="Validation loss"))
            fig_loss.update_layout(title="Evolution de la loss",
                               xaxis_title="Epochs",
                               yaxis_title="Loss",
                               paper_bgcolor="#E8F0F8",
                               plot_bgcolor="#E8F0F8",
                               legend=dict(x=0.98, y=0.98, xanchor="right",yanchor="top"))
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with cols[1]:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"],
                                      mode="lines+markers", name="Train accuracy"))
        
            fig_acc.add_trace(go.Scatter(x=df["epoch"], y=df["val_accuracy"],
                                      mode="lines+markers", name="Validation accuracy"))
            fig_acc.update_layout(title="Evolution de l'accuracy",
                               xaxis_title="Epochs",
                               yaxis_title="Loss",
                               paper_bgcolor="#E8F0F8",
                               plot_bgcolor="#E8F0F8",
                               legend=dict(x=0.98, y=0.02, xanchor="right",yanchor="bottom"))
            st.plotly_chart(fig_acc, use_container_width=True)

        st.subheader("Fine-tuning du modèle")
        st.write("""
                Dans un second temps, les 20 dernières couches du modèle ont été dégelées afin de permettre un ajustement
                 plus fin des représentations apprises et de spécialiser d'avantage le modèle à la tâche de classification.
                 Cette phase de fine-tuning vise à améliorer les performances en autorisant une adaptation partielle
                 des couches profondes aux caratéristiques spécifiques des radiographies thoraciques.
                Les paramètres retenus pour l'entraînement de cette phase sont les suivants:
                - **Optimiseur** : SGD (_learning rate_ = 1.10<sup>-4</sup> et momentum=0.9)
                - **Fonction de perte** : "_Categorical crossentropy_"
                - **Métrique** : accuracy
                - **Callbacks** : _EarlyStopping_, _ModelCheckpoint_ et _ReduceLROnPlateau_.
                 """, unsafe_allow_html=True)
        
        with open("Images/history_inceptionv3_ft_20layers.json", "r") as f:
            history = json.load(f)
        df2 = pd.DataFrame(history)
        df2["epoch"] = range(1, len(df2)+1)

        cols=st.columns(2)
        with cols[0]:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(x=df2["epoch"], y=df2["loss"],
                                      mode="lines+markers", name="Train loss"))
        
            fig_loss.add_trace(go.Scatter(x=df2["epoch"], y=df2["val_loss"],
                                      mode="lines+markers", name="Validation loss"))
            fig_loss.update_layout(title="Evolution de la loss",
                               xaxis_title="Epochs",
                               yaxis_title="Loss",
                               paper_bgcolor="#E8F0F8",
                               plot_bgcolor="#E8F0F8",
                               legend=dict(x=0.98, y=0.98, xanchor="right",yanchor="top"))
            st.plotly_chart(fig_loss, use_container_width=True)
        
        with cols[1]:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=df2["epoch"], y=df2["accuracy"],
                                      mode="lines+markers", name="Train accuracy"))
        
            fig_acc.add_trace(go.Scatter(x=df2["epoch"], y=df2["val_accuracy"],
                                      mode="lines+markers", name="Validation accuracy"))
            fig_acc.update_layout(title="Evolution de l'accuracy",
                               xaxis_title="Epochs",
                               yaxis_title="Loss",
                               paper_bgcolor="#E8F0F8",
                               plot_bgcolor="#E8F0F8",
                               legend=dict(x=0.98, y=0.02, xanchor="right",yanchor="bottom"))
            st.plotly_chart(fig_acc, use_container_width=True)
        st.write("""
                Ces résultats confirment l'intérêt d'un entraînement progressif combinant gel initial des couches
                 et fine-tuning partiel pour exploiter efficacement un modèle pré-entraîné.

                L'écart modéré entre les performances d'entraînement et de validation suggère un léger surapprentissage
                 maîtrisé, confirmant l'éfficacité de la stratégie de fine-tuning adoptée.
                 """)
        

    with onglet3:
        st.header("Evaluation")
        st.write("""
                 L'évaluation sur l'ensemble de test montre des performances globalement équilibrés, avec une accuracy
                 de 84%, confirmant la capacité du modèle à généraliser sur des données non vues.

                 Les métriques de précision, rappel et F1-score sont homogènes entre les classes Sain, Covid-19 et Autres,
                 indiquand un comportement stable du modèle et l'absence de biais marqué en faveur d'une classe
                 spécifique.
                 """)
        data={"Catégories" : ["Sain","Covid-19","Autres"],
              "Précision" : [82,84,87],
              "Rappel" : [88,82,82],
              "F1-Score" : [85,83,84]}
        accuracy = 84

        df = pd.DataFrame(data).melt(id_vars="Catégories",
                                     value_vars=["Précision","Rappel","F1-Score"],
                                     var_name="Métriques",
                                     value_name="Valeur")
        fig = px.bar(df, x="Métriques",
                     y="Valeur",
                     color="Catégories",
                     barmode="group",
                     text="Valeur",
                     title="Rapport de classification durant le test",
                     color_discrete_map={"Sain":"#D070B5",
                                         "Covid-19":"#9A5DAC",
                                         "Autres":"#68AECE"})
        fig.update_traces(texttemplate="%{text:.f}", textposition="outside")
        fig.add_annotation(text=f"<b>Accuracy globale :</b> {accuracy:.1f}%",
                           xref="paper", yref="paper",
                           x=1.3, y=1.1,
                           showarrow=False,
                           font=dict(size=14, color="black"))
        fig.update_layout(yaxis=dict(range=[0,100], title="Score"),
                          xaxis_title="Métriques",
                          legend_title="Catégories",
                          legend=dict(y=0.75),
                          margin=dict(t=80,r=150),
                          bargroupgap=0.05,
                          plot_bgcolor="#E8F0F8",
                          paper_bgcolor="#E8F0F8")
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
                La matrice de confusion met en évidence une majorité de prédictions correctes sur la diagonale,
                 avec des taux de classification supérieurs à 80% pour chaque classe. Les principales confusions
                 concernent les classes Covid-19 et Autres, ce qui s'explique par des similitudes radiologiques,
                 tandis que la classe Sain est globalement bien distinguée.
                 """)

        cm = np.array([[0.8838,0.0686,0.0476],
                       [0.1010,0.8190,0.0800],
                       [0.0914,0.0838,0.8248]])
        labels = ["Sain","Covid-19","Autres"]
        fig_cm = px.imshow(cm*100, x=labels, y=labels,
                           text_auto=".1f",
                           color_continuous_scale="PuBu",
                           labels=dict(x="Classe prédite", y="Classe réelle", color="Score"))
        fig_cm.update_layout(title="Matrice de confusion",
                             xaxis_side="top",
                             height=400,
                             width=400,
                             margin=dict(l=30,r=30,t=60,b=30),
                             paper_bgcolor="#E8F0F8",
                             plot_bgcolor="#E8F0F8")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.write("""
                Dans l'ensemble, ces résultats montrent que le modèle atteint un bon compromis entre performance et robustesse,
                 tout en laissant entrevoir des pistes d'amélioration, notamment sur la discrimination entre les pathologies proches.

                 Des expérimentations complémentaires sont actuellement en cours afin d'améliorer les performances du modèle, notamment
                 à travers la comparaison de différents optimiseurs (SGD, Adam) ainsi que l'évaluation de l'impact de l'application
                 du masque, avec et sans sergmentation préalable.
                 """)