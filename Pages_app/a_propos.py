import streamlit as st
from pathlib import Path
import base64
import plotly.express as px
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go

root_dir = Path(__file__).resolve().parent.parent
def img_to_base64(rel_path):
    with open(root_dir / rel_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

linkedin = img_to_base64("Images/linkedin.png")
github = img_to_base64("Images/github.png")
malt = img_to_base64("Images/malt.png")

def run():
    st.title("Profil & parcours")
    st.write("""
            De formation scientifique, j'ai développé rigueur, curiosité 
            et goût pour la résolution de problématiques complexes lors de mon parcours de chercheuse en chimie. 
            J'applique aujourd'hui ces compétences au domaine innovant et porteur de la data science, 
            au service de l'analyse et de la prise de décision.
            N'hésitez pas à me contacter.
             """)
    cols = st.columns(3)
    with cols[0]:
        st.markdown(f"""
                    <div style="display: flex; gap: 100 px; align-items: center;">
                    <a href="http://www.linkedin.com/in/asma-kerkache" target="_blank">
                    <img src="data:image/png;base64, {linkedin}" width="100">
                    </a>
                    </div>      
                    """, unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown(f"""
                    <div style="display: flex; gap: 100 px; align-items: center;">
                    <a href="http://www.github.com/AsmaSima" target="_blank">
                    <img src="data:image/png;base64, {github}" width="100">
                    </a>
                    </div>
                    """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
                    <div style="display: flex; gap: 100 px; align-items: center;">
                    <a href="http://www.malt.com/profile/asmak" target="_blank">
                    <img src="data:image/png;base64, {malt}" width="100">
                    </a>
                    </div>
                    """, unsafe_allow_html=True)
        
    pdf_path = Path(__file__).parent.parent/"Images"/"Asma KERKACHE_CV_2025.pdf"

    with open(pdf_path, "rb") as f:
        pdf_file = f.read()
    b64 = base64.b64encode(pdf_file).decode()
    cols=st.columns(3)
    with cols[1]:
        st.markdown(f"""
                    <div style="margin-top:30px;">
                    <a href="data.application/pdf;base64,{b64}" 
                    download="Asma_KERKACHE_CV.pdf"
                    style="text-decoration:none;
                    font-size:14px; color:#3A7CA5;">
                    Télécharger mon CV (pdf)
                    </a>
                    </div>
                    """, unsafe_allow_html=True)        
    st.markdown("<hr style='margin-top:5px; margin-bottom:40px;'>", unsafe_allow_html=True)
    
    #----------COMPETENCES----------#
    def run():
        st.markdown("""
                    <style>
                    /* Scope multiselect */
                    div[data-testid="stMultiSelect"] {
                    /* Pastille (container reel) */
                    span[data-baseweb="tag"]{
                    background-color: #D0E2F2 !important;
                    border: 1px solid #D0E2F2 !important;
                    border-radius: 999px !important;}
                    /* Fond interne BaseWeb */
                    span[data-baseweb="tag"]::before,
                    span[data-baseweb="tag"]::after {
                    backgroud-color: #E8F0F8 !important;}
                    /* Texte */
                    span[data-baseweb="tag"] span[title] {
                    color: #1C283B !important;
                    font-weight: 500 !important;}
                    /* icone croix */
                    span[data-baseweb="tag"] svg,
                    span[data-baseweb="tag"] svg * {
                    fill: #1C283B !important;}}
                    /* Conteneur multiselect */
                    div[data-testid="stMultiSelect"]
                    div[data-baseweb="select"] > div {
                    background-color: #E8F0F8 !important;
                    border: 1px solid #D0E2F2 !important;
                    border-radius: 8px !important;}
                    </style>
                    """, unsafe_allow_html=True)
    run()
    skills = [{"label":"Python", "level":4, "category":"Technique"},
              {"label":"Tensorflow", "level":3, "category":"Technique"},
              {"label": "PyTorch", "level":2, "category":"Technique"},
              {"label":"Streamlit", "level":4, "category":"Technique"},
              {"label":"GitHub", "level":4, "category":"Technique"},
              {"label":"HuggingFace", "level":4, "category":"Technique"},
              {"label":"Analyse de données", "level":5, "category":"Scientifique"},
              {"label":"Synthèse organique", "level":4, "category":"Scientifique"},
              {"label":"Synthèse inorganique", "level":4, "category":"Scientifique"},
              {"label":"Rigueur", "level":5, "category":"Savoir-être"},
              {"label":"Curiosité", "level":4, "category":"Savoir-être"},
              {"label":"Autonomie", "level":5, "category":"Savoir-être"},
              {"label":"Communication scientifique", "level":4, "category":"Savoir-être"}]
    
    cols = st.columns(3)
    with cols[1]:
        st.subheader("Compétences")
    all_cats = sorted({s["category"] for s in skills})
    selected_cats = st.multiselect("Filtrer par catégorie", options=all_cats, default=all_cats)

    filtered = [s for s in skills if s["category"] in selected_cats]

    if not filtered:
        st.info("Veuillez sélectionner une ou plusieurs catégorie(s).")
    else:
        level_to_size={1:20, 2:40, 3:50, 4:60, 5:80}
        freq={}
        level_by_word={}
        for s in filtered:
            word = s["label"]
            level = int(s["level"])

            freq[word] = level_to_size[level]
            level_by_word[word] = level
        
        cmap = cm.get_cmap("RdPu")
        def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            lvl = level_by_word.get(word, 3)
            t = (lvl-1)/4.0
            r, g, b, _= cmap(t)
            return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        
        bg = "#E8F0F8"
        wc = WordCloud(width=1100,
                       height=450,
                       background_color=bg,
                       prefer_horizontal=0.95,
                       relative_scaling=0,
                       collocations=False,
                       min_font_size=10,
                       max_words=120,
                       random_state=213).generate_from_frequencies(freq)
        wc = wc.recolor(color_func=color_func, random_state=213)

        img_wc = wc.to_image()
        st.image(img_wc, use_container_width=True)
        
        st.caption("Couleur : Niveau de maitrise (1/clair → 5/foncé).")
        
    #----------PARCOURS----------#
    cols = st.columns(3)
    with cols[1]:
        st.subheader("Parcours")
    
    def add_violin_period(fig, start, end, y, category, title, color_hex, desc="", amp=0.18, n=120, sharp=1.8):
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        xs = pd.date_range(start, end, periods=n)

        t = np.linspace(-1,1,n)
        w = np.clip(1-np.abs(t)**sharp, 0,1)

        y_top = y+amp*w
        y_bot = y-amp*w

        x_poly = np.concatenate([xs.values, xs.values[::-1]])
        y_poly = np.concatenate([y_top, y_bot[::-1]])

        hover_txt = (f"<i>{category}</i><br>"
                     f"<b>{title}</b><br>"
                     f"{start.date()} → {end.date()}<br>"
                     f"{desc}")

        fig.add_trace(go.Scatter(x = pd.to_datetime(x_poly),
                                y = y_poly,
                                name=hover_txt,
                                line = dict(color=color_hex, width=1),
                                fill = "toself",
                                fillcolor = color_hex,
                                opacity = 0.45,
                                showlegend = False))
        x_center = start+(end-start)/2
        title_short = title if len(title)<=15 else title[:15]+"..."
        fig.add_annotation(x=x_center,
                           y=y,
                           text=title_short,
                           showarrow=False,
                           xanchor="center",
                           yanchor="middle",
                           font=dict(size=11, color=color_hex),
                           opacity=0.95)

    data = [{"start": "2024-07-28",
             "end": "2024-10-17",
             "category": "Expérience",
             "titre": "Assistante administrative",
             "details": "● Gestion administrative<br> ● Suivi des préstations<br> ● Organisation"},
             {"start": "2018-10-01",
             "end": "2021-12-17",
             "category": "Expérience",
             "titre": "Chercheuse en chimie",
             "details": "● Collecte de données expérimentales<br> ● Traitement des données expérimentales<br> ● Présentation des résultats<br> ● Gestion de projet<br> ● Encadrement de stagiaires<br> ● Gestion des déchets chimiques"},
             {"start": "2018-03-01",
             "end": "2018-07-06",
             "category": "Expérience",
             "titre": "Ingénieure de recherche",
             "details":"● Synthèse de liquides ioniques à base de synthons biosourcés<br> ● Dissolution de fibres naturelles dans les liquides ioniques"},
             {"start":"2025-09-09",
              "end":"2026-01-31",
              "titre":"Projet : Détection du Covid-19",
              "category":"Expérience",
              "details":"● Deep learning (Tensorflow / PyTorch)<br> ● Data viz'<br> ● Streamlit<br> ● HuggingFace<br> ● GitHub"},
             {"start": "2018-10-01",
             "end": "2021-12-17",
             "category": "Formation",
             "titre": "Doctorat en chimie",
             "details": "● Synthèse organique<br> ● Synthèse inorganique<br> ● Complexes de lanthanide<br> ● Magnétisme"},
             {"start":"2016-09-03",
              "end":"2018-07-03",
              "category":"Formation",
              "titre":"Master Chimie, substances naturelles et médicaments",
              "details":""},
              {"start":"2025-09-09",
              "end":"2025-12-09",
              "category":"Formation",
              "titre":"Data scientist",
              "details":"● Langage Python, bash<br> ● Data viz'<br> ● Machine Learning<br> ● Deep Learning<br> ● LLMs<br> ● Data engineering<br> ● MLOps"}]
    
    df = pd.DataFrame(data)
    colors = {"Formation":"#6C63FF",
            "Expérience":"#FF6584"}
    y_map = {"Formation":1, "Expérience":0}

    cats = st.multiselect("Filtrer par catégorie", sorted(df["category"].unique()),
                         default=sorted(df["category"].unique()))
    df2 = df[df["category"].isin(cats)].sort_values("start")

    if not cats:
        st.info("Veuiller sélectionner une ou plusieurs catégorie(s).")

    fig = go.Figure()
    for _,r in df2.iterrows():
        add_violin_period(fig,
                        r["start"], r["end"],
                        y = y_map[r["category"]],
                        category=r["category"],
                        title=r["titre"],
                        color_hex = colors[r["category"]],
                        desc = r["details"])
    fig.update_layout(height = 320,
                    margin = dict(l=20,r=20,t=60,b=30),
                    xaxis = dict(showgrid=True, gridcolor="#E6E6E6"),
                    yaxis = dict(tickmode="array",
                                tickvals = [1,0],
                                ticktext = ["Formation", "Expérience"],
                                showgrid = False,
                                zeroline = False))

    st.plotly_chart(fig, use_container_width = True)

    #----------PUBLICATIONS ET COMMUNICATIONS----------#

    cols = st.columns([1,3,1])
    with cols[1]:
        st.subheader("Publications & Communications")

    st.write("##### [Journal of Solution chemistry | 2022](https://link.springer.com/article/10.1007/s10953-022-01148-0)")    
    st.image("Images/article.png")

    st.write("##### [Manuscrit de thèse | 2021](https://theses.hal.science/tel-03783573v1/document)")    
    st.image("Images/these.png")

    #----------Centre d'intérêts----------#
    cols=st.columns(3)
    with cols[1]:
        st.subheader("Centres d'intérêts")
    
    cols=st.columns([1,3,1])
    with cols[1]:
        st.write("##### Astrophysique")    
        st.write("Observation planétaire, quantique")

        st.write("##### Photographie argentique")    
        st.write("Couleur, scènes du quotidien/paysages, processus")

        st.write("##### Lecture")    
        st.write("Thriller (Karine Giebel, Donato Carrisi,...)")



