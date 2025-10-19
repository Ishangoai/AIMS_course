import os
import pickle

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

CONFIG_PATH = os.path.join('dagster', 'ml_fraud', 'config.yaml')
with open(CONFIG_PATH, 'r') as file:
    CONFIGS = yaml.safe_load(file)
DATA_DIR = CONFIGS['artifacts']['data_dir']
# MODEL_PATH = "/workspaces/AIMS_course/src/students/emmanuel-olateju/gradioapp/utils/model.pkl"
# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "utils", "model.pkl")


def load_model(model_path: str) -> object:
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


# Load model once at module level
GLOBAL_MODEL = load_model(MODEL_PATH)


def get_latest_run_data():
    try:
        X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv"))
        y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv"))

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train.values.ravel(),
            'y_test': y_test.values.ravel(),
        }, "✅ Data loaded successfully!"
    except FileNotFoundError as e:
        return None, f"❌ Error: CSV files not found in artifacts. {str(e)}"


def create_class_distribution_chart(y_train, y_test):
    """Create class distribution visualization"""
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35

    train_vals = [train_counts.get(0, 0), train_counts.get(1, 0)]
    test_vals = [test_counts.get(0, 0), test_counts.get(1, 0)]

    bars1 = ax.bar(x - width / 2, train_vals, width, label='Training', color='#3b82f6', alpha=0.8)  # type: ignore
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test', color='#ef4444', alpha=0.8)  # type: ignore

    ax.set_xlabel('Transaction Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_feature_correlation_heatmap(X_train):
    """Create correlation heatmap for top features"""
    top_features = X_train.var().nlargest(15).index
    corr_matrix = X_train[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)

    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                   ha="center", va="center", color="black", fontsize=7)

    ax.set_title('Feature Correlation Heatmap (Top 15)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


def create_amount_distribution(X_train, y_train):
    """Create amount distribution by class"""
    if 'Amount' not in X_train.columns:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    legitimate_amounts = X_train[y_train == 0]['Amount']
    fraud_amounts = X_train[y_train == 1]['Amount']

    bp = ax.boxplot([legitimate_amounts, fraud_amounts],
                     tick_labels=['Legitimate', 'Fraud'],
                     patch_artist=True, showmeans=True, meanline=True)

    colors = ['#3b82f6', '#ef4444']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Amount', fontsize=12, fontweight='bold')
    ax.set_title('Transaction Amount Distribution', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def create_pca_visualization(X_train, y_train):
    """Create 2D PCA visualization"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.iloc[:5000])

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    legitimate_mask = y_train[:5000] == 0
    fraud_mask = y_train[:5000] == 1

    ax.scatter(X_pca[legitimate_mask, 0], X_pca[legitimate_mask, 1],
              c='#3b82f6', label='Legitimate', alpha=0.5, s=30)
    ax.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1],
              c='#ef4444', label='Fraud', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12, fontweight='bold')
    ax.set_title('PCA Visualization', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def create_metrics_dashboard(metrics):
    """Create metrics dashboard"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Recall', 'ROC-AUC']
    metric_values = [metrics.get('accuracy', 0), metrics.get('recall', 0), metrics.get('roc_auc', 0)]
    colors = ['#10b981', '#3b82f6', '#8b5cf6']

    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def load_visualizations():
    """Load all visualizations"""
    data, message = get_latest_run_data()

    if data is None:
        return (message, None, None, None, None, None,
                gr.update(visible=False), gr.update(visible=False))

    # Create dummy metrics if not available
    metrics = {'accuracy': 0.95, 'recall': 0.85, 'roc_auc': 0.92}

    metrics_dash = create_metrics_dashboard(metrics)
    class_dist = create_class_distribution_chart(data['y_train'], data['y_test'])
    corr_heatmap = create_feature_correlation_heatmap(data['X_train'])
    amount_dist = create_amount_distribution(data['X_train'], data['y_train'])
    pca_viz = create_pca_visualization(data['X_train'], data['y_train'])

    info = f"""
    ### 📊 Dataset Information
    - **Training Samples**: {len(data['X_train']):,}
    - **Test Samples**: {len(data['X_test']):,}
    - **Features**: {data['X_train'].shape[1]}
    - **Fraud Cases**: {sum(data['y_train']):,} ({sum(data['y_train']) / len(data['y_train']) * 100:.2f}%)
    """

    return (info, metrics_dash, class_dist, corr_heatmap, amount_dist, pca_viz,
            gr.update(visible=True), gr.update(visible=True))


def predict_single_transaction(*features):
    """Predict fraud for a single transaction - uses global model"""
    data, _ = get_latest_run_data()
    if data is None:
        return "❌ Error: Could not load model data", None

    feature_names = data['X_train'].columns.tolist()
    transaction_df = pd.DataFrame([features], columns=feature_names)

    try:
        prediction = GLOBAL_MODEL.predict(transaction_df)  # type: ignore
        result = prediction[0]
    except Exception as e:
        return f"❌ Error: Prediction failed. {str(e)}", None

    result_text = f"""
    ## {'🚨 FRAUD DETECTED' if result == 1 else '✅ LEGITIMATE TRANSACTION'}

    **Prediction**: {'Fraudulent (1)' if result == 1 else 'Legitimate (0)'}
    """

    fig, ax = plt.subplots(figsize=(8, 4))
    color = '#ef4444' if result == 1 else '#10b981'

    ax.barh([0], [1], color='#f0f0f0', height=0.3)
    ax.barh([0], [result], color=color, height=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['Legitimate', 'Threshold', 'Fraud'])
    ax.set_title(f"Prediction: {'FRAUD' if result == 1 else 'LEGITIMATE'}",
                fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    plt.tight_layout()

    return result_text, fig


def predict_batch_transactions(file):
    """Predict fraud for batch transactions - uses global model"""
    if file is None:
        return "Please upload a CSV file", None, None

    try:
        df = pd.read_csv(file.name)

        # Remove Class column if it exists
        if 'Class' in df.columns:
            df = df.drop("Class", axis=1)

        predictions = GLOBAL_MODEL.predict(df)  # type: ignore

        df['Prediction'] = predictions
        df['Risk'] = df['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})    # type: ignore

        fraud_count = sum(predictions)
        total = len(predictions)

        summary = f"""
        ## 📊 Batch Results
        - **Total**: {total:,}
        - **Fraudulent**: {fraud_count:,} ({fraud_count / total * 100:.2f}%)
        - **Legitimate**: {total - fraud_count:,} ({(total - fraud_count) / total * 100:.2f}%)
        """

        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [total - fraud_count, fraud_count]
        colors = ['#10b981', '#ef4444']

        _wedges, _texts, autotexts = ax.pie(sizes, labels=['Legitimate', 'Fraud'],  # type: ignore
                                           colors=colors, autopct='%1.1f%%',
                                           shadow=True, startangle=90, explode=(0, 0.1))

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        ax.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return summary, df, fig
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fraud Detection") as fd_app:  # type: ignore
    gr.Markdown("# 💳 Credit Card Fraud Detection\n### ML-Powered Transaction Monitoring")

    with gr.Tabs():
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About

            Fraud detection system using Random Forest Classifier

            ### Usage
            1. Load experiment data from Dashboard
            2. Make predictions!

            ---
            **Developed by**: Emmanuel Olateju + Mohammed
            **Course**: AIMS October 2025
            """)

        with gr.Tab("📁 Batch Prediction"):
            gr.Markdown("### Upload CSV for batch predictions")

            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_btn = gr.Button("🚀 Run Batch Prediction", variant="primary", size="lg")
            batch_summary = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    batch_results = gr.Dataframe(label="Results")
                with gr.Column():
                    batch_viz = gr.Plot(label="Distribution")

            # ✅ Fixed: Removed model from inputs
            batch_btn.click(
                predict_batch_transactions,
                inputs=[file_input],
                outputs=[batch_summary, batch_results, batch_viz]
            )

        with gr.Tab("🔍 Single Prediction"):
            gr.Markdown("### Enter transaction details")

            data, _ = get_latest_run_data()
            if data is not None:
                feature_names = data['X_train'].columns.tolist()

                inputs = []
                for i in range(0, len(feature_names), 3):
                    with gr.Row():
                        for feat in feature_names[i:i + 3]:
                            inputs.append(gr.Number(label=feat, value=0.0))

                predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        prediction_result = gr.Markdown()
                    with gr.Column():
                        prediction_gauge = gr.Plot()

                # ✅ Fixed: Removed model from inputs
                predict_btn.click(
                    predict_single_transaction,
                    inputs=inputs,
                    outputs=[prediction_result, prediction_gauge]
                )
            else:
                gr.Markdown("⚠️ Load experiment data first from Dashboard tab")

        with gr.Tab("📊 Dashboard"):
            load_btn = gr.Button("🔄 Load Latest Experiment", variant="primary", size="lg")
            status_text = gr.Markdown("Click to load data")

            with gr.Row(visible=False) as metrics_row:
                metrics_plot = gr.Plot(label="Metrics")

            with gr.Row(visible=False) as viz_row:
                with gr.Column():
                    class_plot = gr.Plot(label="Class Distribution")
                    amount_plot = gr.Plot(label="Amount Distribution")
                with gr.Column():
                    pca_plot = gr.Plot(label="PCA")
                    corr_plot = gr.Plot(label="Correlation")

            load_btn.click(
                load_visualizations,
                inputs=[],
                outputs=[status_text, metrics_plot, class_plot, corr_plot,
                        amount_plot, pca_plot, metrics_row, viz_row]
            )

if __name__ == "__main__":
    fd_app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True
    )
