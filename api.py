import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# === CONFIGURATION DE LA CLÉ API DEEPSEEK ===
os.environ["DEEPSEEK_API_KEY"] = "sk-3b5fac139a8d428c8cd9fc3f3b96be2f"  # Remplacez par votre vraie clé
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# === INITIALISATION DU MODÈLE DE CHAT ===
llm = ChatOpenAI(
    temperature=0.2,
    model_name="deepseek-chat",  # Vérifiez le nom exact du modèle dans la documentation
    openai_api_base=DEEPSEEK_API_BASE,
    openai_api_key=os.environ["DEEPSEEK_API_KEY"]
)

# === TEMPLATE DE PROMPT (identique) ===
prompt = PromptTemplate(
    input_variables=["description", "cv"],
    template="""
Tu es un expert RH. Analyse ce CV par rapport à la description suivante :

DESCRIPTION DU POSTE :
----------------------
{description}

CV DU CANDIDAT :
----------------
{cv}

Est-ce que ce candidat est adapté au poste ? Réponds par "Oui" ou "Non", avec une justification courte (2-3 phrases).
"""
)

# ... (le reste du code reste identique à partir d'ici)

# === CRÉATION DE LA CHAÎNE ===
chaine = LLMChain(llm=llm, prompt=prompt)

# === LIRE LES FICHIERS ===
def lire_fichier(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        return f.read()

# === LECTURE DES FICHIERS DU DOSSIER COURANT ===
dossier = os.getcwd()
description_poste = lire_fichier(os.path.join(dossier, "description_poste.txt"))

# === CHARGER LES CVS ===
liste_cvs = {}
for fichier in os.listdir(dossier):
    if fichier.endswith(".txt") and fichier != "description_poste.txt":
        liste_cvs[fichier] = lire_fichier(os.path.join(dossier, fichier))

# === ANALYSE ===
print("Résultats d’analyse :\n")
for nom, cv in liste_cvs.items():
    print(f" {nom}")
    resultat = chaine.invoke({"description": description_poste, "cv": cv})
    print(resultat["text"])
    print("-" * 80)