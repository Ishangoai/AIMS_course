import tempfile

import gradio as gr
import matplotlib
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Utiliser le backend 'Agg' qui est non-interactif et adapté pour les serveurs
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =========================
# Configuration and Model Loading
# =========================
# Charger le modèle MLflow depuis l'URI spécifié
MODEL_URI = "runs:/71360b81333e42de88cc64e27caad3d2/model"
try:
    fraud_model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    print(f"Error loading model: {e}")
    # Créer un modèle factice pour permettre à l'interface de se lancer même sans le vrai modèle
    class DummyModel:
        def predict(self, df):
            # Retourne des prédictions aléatoires (0 ou 1)
            return np.random.randint(0, 2, size=len(df))
    fraud_model = DummyModel()
    print("Using a dummy model.")


# Charger le jeu de données par défaut pour obtenir les noms et les plages des fonctionnalités
DATA_URL = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
default_df = pd.read_csv(DATA_URL)

# Extraire les noms des fonctionnalités (toutes les colonnes sauf 'Class')
feature_names = [c for c in default_df.columns if c != "Class"]
# Calculer les plages (min, max) pour chaque fonctionnalité pour les curseurs
feature_ranges = {col: (default_df[col].min(), default_df[col].max()) for col in feature_names}


# =========================
# Prediction Functions
# =========================

# 1. Batch Prediction Function
def predict_batch(file, use_default):
    """Exécute la prédiction sur un fichier CSV entier."""
    if use_default:
        df = default_df.copy()
    elif file is not None:
        df = pd.read_csv(file.name)
    else:
        # Retourner des objets vides si aucune donnée n'est fournie
        empty_fig = plt.figure()
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, None

    features_to_predict = df[feature_names]

    # Faire des prédictions
    predictions = fraud_model.predict(features_to_predict)
    df["Fraud_Prediction"] = predictions
    df["Fraud_Prediction"] = df["Fraud_Prediction"].map({0: "Non-Fraud", 1: "Fraud"})

    # --- Créer les visualisations avec Matplotlib ---

    # Histogramme des montants
    fig_amount_hist, ax1 = plt.subplots()
    fraud_amounts = df[df['Fraud_Prediction'] == 'Fraud']['Amount']
    non_fraud_amounts = df[df['Fraud_Prediction'] == 'Non-Fraud']['Amount']
    ax1.hist(non_fraud_amounts, bins=50, alpha=0.7, label='Non-Fraud', color='#4CAF50')
    ax1.hist(fraud_amounts, bins=50, alpha=0.7, label='Fraud', color='#F44336')
    ax1.set_title('Distribution of Amounts by Fraud Prediction')
    ax1.set_xlabel('Amount')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    plt.tight_layout()

    # Diagramme circulaire des fraudes
    fig_fraud_pie, ax2 = plt.subplots()
    proportions = df['Fraud_Prediction'].value_counts()
    ax2.pie(proportions, labels=proportions.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
    ax2.set_title('Fraud / Non-Fraud Proportion')
    ax2.axis('equal')  # Assure que le diagramme est un cercle

    # Fonction d'aide pour créer des histogrammes de fonctionnalités
    def create_feature_hist(df_data, feature_name):
        fig, ax = plt.subplots()
        fraud_data = df_data[df_data['Fraud_Prediction'] == 'Fraud'][feature_name]
        non_fraud_data = df_data[df_data['Fraud_Prediction'] == 'Non-Fraud'][feature_name]
        ax.hist(non_fraud_data, bins=50, alpha=0.7, label='Non-Fraud', color='#4CAF50')
        ax.hist(fraud_data, bins=50, alpha=0.7, label='Fraud', color='#F44336')
        ax.set_title(f'Distribution of {feature_name}')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
        return fig

    fig_feature1 = create_feature_hist(df, feature_names[0])
    fig_feature2 = create_feature_hist(df, feature_names[1])
    fig_feature3 = create_feature_hist(df, feature_names[2])

    # Convertir le DataFrame en CSV pour le téléchargement
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        return fig_amount_hist, fig_fraud_pie, fig_feature1, fig_feature2, fig_feature3, tmp_file.name


# 2. Single Prediction Function
def predict_single(*slider_values):
    """
    Exécute la prédiction pour un ensemble unique de valeurs de fonctionnalités.
    """
    # Créer un DataFrame à partir des valeurs des curseurs
    input_data = {name: val for name, val in zip(feature_names, slider_values)}
    input_df = pd.DataFrame([input_data])

    # Obtenir la prédiction (0 ou 1)
    prediction_val = fraud_model.predict(input_df)[0]

    # Tenter d'obtenir la probabilité réelle si le modèle le supporte
    try:
        if hasattr(fraud_model, "predict_proba"):
            probabilities = fraud_model.predict_proba(input_df)
            probability = probabilities[0][1]  # Probabilité de la classe positive 'Fraude'
        elif hasattr(fraud_model.python_model, "predict_proba"):
            probabilities = fraud_model.python_model.predict_proba(input_df)
            probability = probabilities[0][1]
        else:
            raise AttributeError("Model does not have a predict_proba method.")
    except (AttributeError, TypeError, NotImplementedError):
        print("Warning: Using a simulated probability for visualization.")
        probability = 0.85 if prediction_val == 1 else 0.15

    prediction_text = "FRAUD" if prediction_val == 1 else "Not Fraud"
    color = "#F44336" if prediction_val == 1 else "#4CAF50"
    probability_percent = probability * 100

    # Créer un graphique "jauge" avec Matplotlib (en utilisant un diagramme à barres horizontal)
    fig, ax = plt.subplots(figsize=(6, 1), facecolor='#F9F9F9')
    fig.patch.set_facecolor('#F9F9F9')

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)

    # Barre de fond
    ax.barh([0.5], [100], color='lightgray', height=0.5, edgecolor='gray')
    # Barre de premier plan
    ax.barh([0.5], [probability_percent], color=color, height=0.5)

    # Texte de la probabilité
    text_ha = 'left' if probability_percent < 90 else 'right'
    ax.text(
        probability_percent + 2,
        0.5,
        f'{probability_percent:.1f}%',
        va='center',
        ha=text_ha,
        fontsize=12,
        weight='bold'
    )
    ax.set_title("Fraud Probability (%)", fontsize=14, pad=10)

    # Nettoyer l'esthétique
    ax.set_axis_off()
    plt.tight_layout()

    result_label = gr.Label(value=prediction_text, label="Prediction Result")
    return fig, result_label


# =========================
# Gradio Interface
# =========================
with gr.Blocks(theme=gr.themes.Soft(), title="Fraud Detection") as appli_fraud:
    gr.Markdown("# 💳 Credit Card Fraud Detection")
    gr.Markdown(
        "Choose a prediction mode: `Single Prediction` to test one transaction "
        "or `Batch Prediction` to analyze a CSV file."
    )

    with gr.Tabs():
        # Onglet 1: Single Prediction
        with gr.TabItem("Single Prediction"):
            with gr.Row():
                # Création dynamique des colonnes de curseurs
                all_sliders = []
                num_features_per_col = 10
                for i in range(0, len(feature_names), num_features_per_col):
                    with gr.Column(scale=2):
                        col_features = feature_names[i:i + num_features_per_col]
                        gr.Markdown(f"#### Features {i + 1} to {i + len(col_features)}")
                        for feature in col_features:
                            min_val, max_val = feature_ranges[feature]
                            slider = gr.Slider(
                                minimum=min_val,
                                maximum=max_val,
                                value=np.mean([min_val, max_val]),
                                label=feature,
                                interactive=True,
                                step=abs(max_val - min_val) / 1000
                            )
                            all_sliders.append(slider)

                # Colonne pour les résultats de la prédiction
                with gr.Column(scale=3):
                    gr.Markdown("#### 🎯 Prediction Result")
                    predict_single_button = gr.Button("Run Prediction", variant="primary")
                    output_gauge = gr.Plot(label="Probability Gauge")
                    output_label = gr.Label(label="Result")

            predict_single_button.click(
                fn=predict_single,
                inputs=all_sliders,
                outputs=[output_gauge, output_label]
            )

        # Onglet 2: Batch Prediction
        with gr.TabItem("Batch Prediction"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Input Options")
                    upload_file = gr.File(label="Upload your CSV file", file_types=[".csv"])
                    use_default_checkbox = gr.Checkbox(label="Use default dataset", value=True)
                    predict_batch_button = gr.Button("Run Predictions", variant="primary")
                    download_file = gr.File(label="Download CSV with predictions")

                with gr.Column(scale=2):
                    gr.Markdown("#### Results Visualizations")
                    with gr.Tabs():
                        with gr.TabItem("Amount"):
                            output_amount_hist = gr.Plot()
                        with gr.TabItem("Proportion"):
                            output_fraud_pie = gr.Plot()
                        with gr.TabItem(f"{feature_names[0]}"):
                            output_feature1 = gr.Plot()
                        with gr.TabItem(f"{feature_names[1]}"):
                            output_feature2 = gr.Plot()
                        with gr.TabItem(f"{feature_names[2]}"):
                            output_feature3 = gr.Plot()

            predict_batch_button.click(
                fn=predict_batch,
                inputs=[upload_file, use_default_checkbox],
                outputs=[
                    output_amount_hist,
                    output_fraud_pie,
                    output_feature1,
                    output_feature2,
                    output_feature3,
                    download_file
                ]
            )

# Lancer l'application
appli_fraud.launch(share=True)
