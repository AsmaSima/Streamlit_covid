import streamlit as st

def run():
    st.title("Détection du Covid-19 sur des radiographies pulmonaires")
    st.image("Images/accueil.jpeg", width=500)
    st.write("""
            Cette application a été développée dans le cadre d'un projet DataScientest.
            Elle vise à entraîner un modèle de deep learning capable d'identifier des patients atteints
            de Covid-19 à partir de radiographies thoraciques.

            Plusieurs entraînements ont été réalisés afin de comparer différentes approches (pré-traitement, masquage
             automatique, choix des modèles et des paramètres).

            Utilisez le menu latéral pour accéder aux différentes sections :
            - L'exploration du jeu de données
            - La modélisation du masquage automatique
            - La modélisation de classification
            - Tester le modèle
            - Présentation de mon profil et mon parcours
            """)
    st.write("""
             Le modèle est en cours d'amélioration continue. Les performances présentées correspondent
             à l'état actuel des expérimentations.
             """)
    
    st.write("""
             **Avertissement** : Cette application est fournie à des fins pédagogiques et expérimentales uniquement.
             Elle ne constitue en aucun cas un outil de diagnostic médical et ne doit pas se substituer à l'avis d'un
             professionnel de santé.
             """)