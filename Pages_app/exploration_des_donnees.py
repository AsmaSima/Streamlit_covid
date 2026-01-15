import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import random
from PIL import Image
from pathlib import Path

def run():
    st.title("Exploration des données")
    st.write("""
             Le jeu de données est composé d'images radiographiques et des masques correspondants.
             Les données sont issues de plusieurs sources publiques et regrouper sur Kaggle.
             [<sup>[1]</sup>](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database/data)
             La structure de l'organisation de ces dernière est la suivante :
             """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("Images/structure_data.png",
                caption="Structure du jeu de données",
                width=250)
        
    # ---- Histogramme ----  
    categories = ["Covid-19", "Opacité pulmonaire", "Normal", "Pneumonie virale"]
    nb_img = [3616, 6012, 10192, 1345]
    nb_mask = [3616, 6012, 10192, 1345]

    hist = pd.DataFrame({"Catégories" : categories,
                        "Images" : nb_img,
                        "Masques" : nb_mask})
    hist_long = hist.melt(id_vars = "Catégories",
                          value_vars = ["Images", "Masques"],
                          var_name = "Type",
                          value_name= "Nombre")
    fig = px.bar(hist_long, x="Catégories",
                 y="Nombre",
                 color="Type",
                 barmode="group",
                 title="Répartition des données par catégorie")
    
    fig.update_layout(xaxis_title="Catégories",
                      yaxis_title="Nombre",
                      legend_title="",
                      template="plotly",
                      plot_bgcolor="#E8F0F8",
                      paper_bgcolor="#E8F0F8")
    
    fig.update_traces(marker_line_width=5,
                      text=hist_long["Nombre"],
                      textposition="inside")
    fig.for_each_trace(lambda t: t.update(marker_color="#113A69") if t.name =="Images" else t.update(marker_color="#ce519a"))

    st.subheader("Répartition des données par catégorie")
    st.write("""
            - Disparté dans le nombre d'images par catégorie
            - Nombre d'images et nombre de masques égaux
            - Pas de fichiers manquants ou corrompu
             """)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Taille et formats des fichiers")
    rows = [categories[:2], categories[2:]]
    for row in rows:
        cols = st.columns(2)
        for col, cat in zip(cols, row):
            with col:
                st.markdown(f"""
                            **{cat}**
                            - Images : PNG - 299x299 px
                            - Masques : PNG - 256x256 px
                            """)
    
    # ---- Figure taille/formats ----
    fig = go.Figure()

    # ----Grand carré----
    fig.add_shape(type="rect",
                  x0=0,
                  y0=0,
                  x1=1,
                  y1=1,
                  line=dict(width=0),
                  fillcolor="rgba(0,63,140,0.85)")
    #----Ombre----
    fig.add_shape(type="rect",
                  x0=0.04, y0=0.04,
                  x1=0.84, y1=0.84,
                  line=dict(width=0),
                  fillcolor="rgba(0,0,0,0.35)")
    #----Petit carré----
    fig.add_shape(type="rect",
                  x0=0.02, y0=0.02,
                  x1=0.82, y1=0.82,
                  line=dict(width=0),
                  fillcolor="rgba(255,122,200,1)")
    # ---- Légende ----
    fig.add_annotation(x=0.5, y=0.92,
                       text="Images<br>299x299 px - PNG",
                       showarrow=False,
                       font=dict(size=16, color="#90b0d6"))
    fig.add_annotation(x=0.4,
                       y=0.4,
                       text="Masques<br>256x256 px - PNG",
                       showarrow=False,
                       font=dict(size=14, color="#7a0040"))

    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_yaxes(visible=False, range=[0,1], scaleanchor="x", scaleratio=1)

    fig.update_layout(height=300,
                      width=300,
                      margin=dict(l=10,r=10,t=10,b=10),
                      plot_bgcolor="#E8F0F8",
                      paper_bgcolor="#E8F0F8")
    st.plotly_chart(fig, use_container_width=True)

    # ---- Echantillon d'images ----
    def charger_img(base_dir):
        """
        base_dir doit contenir 3 sous-dossiers
        """
        classes = ["COVID", "Sain", "Autres"]
        data ={}
        for cls in classes:
            dossier = os.path.join(base_dir, cls, "images")
            if os.path.join(dossier):
                fichiers = [os.path.join(dossier,f) for f in os.listdir(dossier)]
                data[cls] = fichiers
        return data

    def carrousel_aleatoire(base_dir:str, n=4, exts=(".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
        """
        Selectionne n images alétoires avec leur étiquettes
        """
        base = Path(base_dir)
        files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not files:
            raise FileNotFoundError(f"Aucune image trouvée dans {base.resolve()}")
        k = min(n,len(files))
        sample = random.sample(files, k)       
        
        def infer_label(p: Path) -> str:
            name = p.stem.lower()
            if "covid" in name:
                return "Covid-19"
            if "normal" in name or "sain" in name or "healthy" in name:
                return "Sain"
            return "Autres"
        return [(str(p), infer_label(p)) for p in sample]
    
    tirage = carrousel_aleatoire("Datas", n=4)
    
    if "echantillon" not in st.session_state:
        st.session_state.echantillon = []
    if "idx_img" not in st.session_state:
        st.session_state.idx_img = []

    st.subheader("Visualisation d'un échantillon d'images du dataset")
    if st.button("Générer un échantillon aléatoire"):
        st.session_state.echantillon = tirage
        st.session_state.idx_img = 0

    if st.session_state.echantillon:
        col_prev, col_img, col_next = st.columns([1,3,1])

        with col_prev:
            if st.button("⬅️Précédent"):
                st.session_state.idx_img = (st.session_state.idx_img-1)% len(st.session_state.echantillon)
        with col_next:
            if st.button("➡️Suivant"):
                st.session_state.idx_img = (st.session_state.idx_img+1)% len(st.session_state.echantillon)
        with col_img:
            img_path, cls= st.session_state.echantillon[st.session_state.idx_img]
            img = Image.open(img_path)
            st.image(img, caption=f"Classe {cls}", width=350, use_container_width=True)
            st.markdown(f"<p style='text-align:center;margin-top:O.5rem;'>Image {st.session_state.idx_img+1}/{len(st.session_state.echantillon)}</p>",
                        unsafe_allow_html=True)           
    else:
        st.info("Cliquez sur le bouton pour afficher un échantillon")