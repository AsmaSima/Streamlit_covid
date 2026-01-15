import streamlit as st

from Pages_app.accueil import run as accueil
from Pages_app.exploration_des_donnees import run as exploration
from Pages_app.masque_automatique import run as masque
from Pages_app.modelisation_inceptionv3 import run as modelisation
from Pages_app.tester_le_modele import run as tester
from Pages_app.a_propos import run as apropos

st.set_page_config(page_title="Détection Covid-19", layout="centered")

pages = {"Acceuil" : accueil,
        "Exploration des données" : exploration,
        "Masque automatique" : masque,
        "Modélisation - Inception V3" : modelisation,
        "Tester le modèle" : tester,
        "Profil" : apropos}

choice = st.sidebar.radio("Menu", list(pages.keys()))
pages[choice]()