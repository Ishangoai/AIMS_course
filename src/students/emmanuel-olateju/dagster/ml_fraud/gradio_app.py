import warnings

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import requests
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8-darkgrid')

# MLflow configuration
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_API_ENDPOINT = "http://127.0.0.1:50002/fraud-detection/invocations"
EXPERIMENT_NAME = "fraud_detection_experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_latest_run_data():
    """Retrieve training data from the latest MLflow run"""
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            return None, "Experiment not found. Please run the training pipeline first."

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            return None, "No runs found in the experiment."

        latest_run = runs[0]
        run_id = latest_run.info.run_id

        # Load logged datasets
        artifacts_path = client.download_artifacts(run_id, "")

        # Try to load the datasets from artifacts
        try:
            X_train = pd.read_csv(f"{artifacts_path}/X_train.csv")
            X_test = pd.read_csv(f"{artifacts_path}/X_test.csv")
            y_train = pd.read_csv(f"{artifacts_path}/y_train.csv")
            y_test = pd.read_csv(f"{artifacts_path}/y_test.csv")

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train.values.ravel(),
                'y_test': y_test.values.ravel(),
                'run_id': run_id,
                'metrics': latest_run.data.metrics
            }, "Data loaded successfully!"
        except Exception as e:
            return None, f"Error loading artifacts: {str(e)}"

    except Exception as e:
        return None, f"Error connecting to MLflow: {str(e)}"


def predict_fraud(transaction_data):
    """Make prediction using the model API endpoint"""
    try:
        # Prepare the payload
        payload = {
            "dataframe_split": {
                "columns": transaction_data.columns.tolist(),
                "data": transaction_data.values.tolist()
            }
        }

        # Make API request
        response = requests.post(
            MODEL_API_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            predictions = response.json()['predictions']
            return predictions
        else:
            return None

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


def create_class_distribution_chart(y_train, y_test):
    """Create class distribution visualization"""
    train_counts = pd.Series(y_train).value_counts()
    test_counts = pd.Series(y_test).value_counts()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(2)
    width = 0.35

    train_vals = [train_counts.get(0, 0), train_counts.get(1, 0)]
    test_vals = [test_counts.get(0, 0), test_counts.get(1, 0)]

    bars1 = ax.bar(x - width / 2, train_vals, width, label='Training Set', color='#3b82f6', alpha=0.8)
    bars2 = ax.bar(x + width / 2, test_vals, width, label='Test Set', color='#ef4444', alpha=0.8)

    ax.set_xlabel('Transaction Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Class Distribution in Training and Test Sets', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{int(height):,}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig


def create_feature_correlation_heatmap(X_train):
    """Create correlation heatmap for top features"""
    # Select top 15 features by variance for better visualization
    top_features = X_train.var().nlargest(15).index
    corr_matrix = X_train[top_features].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation', rotation=270, labelpad=20, fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)

    # Add text annotations
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    ax.set_title('Feature Correlation Heatmap (Top 15 Features by Variance)',
                 fontsize=14, fontweight='bold', pad=20)

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
                     patch_artist=True,
                     showmeans=True,
                     meanline=True)

    # Color the boxes
    colors = ['#3b82f6', '#ef4444']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Amount', fontsize=12, fontweight='bold')
    ax.set_title('Transaction Amount Distribution by Class', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def create_pca_visualization(X_train, y_train):
    """Create 2D PCA visualization"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize and apply PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train.iloc[:5000])  # Sample for performance

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Separate by class
    legitimate_mask = y_train[:5000] == 0
    fraud_mask = y_train[:5000] == 1

    ax.scatter(X_pca[legitimate_mask, 0], X_pca[legitimate_mask, 1],
              c='#3b82f6', label='Legitimate', alpha=0.5, s=30, edgecolors='none')
    ax.scatter(X_pca[fraud_mask, 0], X_pca[fraud_mask, 1],
              c='#ef4444', label='Fraud', alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                  fontsize=12, fontweight='bold')
    ax.set_title('PCA Visualization of Transaction Data', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=10, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_metrics_dashboard(metrics):
    """Create metrics dashboard"""
    fig, ax = plt.subplots(figsize=(10, 6))

    metric_names = ['Accuracy', 'Recall', 'ROC-AUC']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('recall', 0),
        metrics.get('roc_auc', 0)
    ]

    colors = ['#10b981', '#3b82f6', '#8b5cf6']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
               f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def load_visualizations():
    """Load all visualizations"""
    data, message = get_latest_run_data()

    if data is None:
        return (message, None, None, None, None, None,
                gr.update(visible=False), gr.update(visible=False))

    # Create visualizations
    metrics_dash = create_metrics_dashboard(data['metrics'])
    class_dist = create_class_distribution_chart(data['y_train'], data['y_test'])
    corr_heatmap = create_feature_correlation_heatmap(data['X_train'])
    amount_dist = create_amount_distribution(data['X_train'], data['y_train'])
    pca_viz = create_pca_visualization(data['X_train'], data['y_train'])

    # Create dataset info
    info = f"""
    ### Dataset Information
    - **Training Samples**: {len(data['X_train']):,}
    - **Test Samples**: {len(data['X_test']):,}
    - **Features**: {data['X_train'].shape[1]}
    - **Fraud Cases (Train)**: {sum(data['y_train']):,} ({sum(data['y_train']) / len(data['y_train']) * 100:.2f}%)
    - **MLflow Run ID**: `{data['run_id']}`
    """

    return (info, metrics_dash, class_dist, corr_heatmap,
            amount_dist, pca_viz,
            gr.update(visible=True), gr.update(visible=True))


def predict_single_transaction(*features):
    """Predict fraud for a single transaction"""
    data, _ = get_latest_run_data()

    if data is None:
        return "❌ Error: Could not load model data", None

    # Get feature names from X_train
    feature_names = data['X_train'].columns.tolist()

    # Create DataFrame with the input
    transaction_df = pd.DataFrame([features], columns=feature_names)

    # Make prediction
    prediction = predict_fraud(transaction_df)

    if prediction is None:
        return "❌ Error: Prediction failed. Check if model server is running.", None

    result = prediction[0]
    confidence = "High Risk" if result == 1 else "Low Risk"

    result_text = f"""
    ## Prediction Result

    {'🚨 **FRAUD DETECTED**' if result == 1 else '✅ **LEGITIMATE TRANSACTION**'}

    **Risk Level**: {confidence}
    **Prediction**: {'Fraudulent (Class 1)' if result == 1 else 'Legitimate (Class 0)'}
    """

    # Create gauge chart
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create a simple visual representation
    color = '#ef4444' if result == 1 else '#10b981'
    label = 'FRAUD' if result == 1 else 'LEGITIMATE'

    # Create a horizontal bar as gauge
    ax.barh([0], [1], color='#f0f0f0', height=0.3)
    ax.barh([0], [result], color=color, height=0.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0 (Legitimate)', '0.5', '1 (Fraud)'], fontsize=10)
    ax.set_title(f'Prediction: {label}', fontsize=14, fontweight='bold', pad=20)

    # Add vertical line at 0.5 threshold
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.text(0.5, 0.4, 'Threshold', ha='center', fontsize=9, color='red')

    # Add result text
    ax.text(result, -0.15, f'{result}', ha='center', fontsize=12, fontweight='bold', color=color)

    plt.tight_layout()
    return result_text, fig


def predict_batch_transactions(file):
    """Predict fraud for batch transactions"""
    if file is None:
        return "Please upload a CSV file", None, None

    try:
        df = pd.read_csv(file.name)

        # Make predictions
        predictions = predict_fraud(df)

        if predictions is None:
            return "❌ Error: Prediction failed", None, None

        df['Prediction'] = predictions
        df['Risk'] = df['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})

        # Summary statistics
        fraud_count = sum(predictions)
        total = len(predictions)

        summary = f"""
        ## Batch Prediction Results

        - **Total Transactions**: {total:,}
        - **Fraudulent**: {fraud_count:,} ({fraud_count / total * 100:.2f}%)
        - **Legitimate**: {total - fraud_count:,} ({(total - fraud_count) / total * 100:.2f}%)
        """

        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))

        sizes = [total - fraud_count, fraud_count]
        labels = ['Legitimate', 'Fraud']
        colors = ['#10b981', '#ef4444']
        explode = (0, 0.1)  # explode fraud slice

        _, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,  # type: ignore
                                           autopct='%1.1f%%', shadow=True, startangle=90)

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')

        ax.set_title('Batch Prediction Distribution', fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return summary, df, fig

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Fraud Detection System") as demo:  # type: ignore
    gr.Markdown("""
    # 💳 Credit Card Fraud Detection System
    ### ML-Powered Transaction Monitoring with MLflow Integration
    """)

    with gr.Tabs():
        # Tab 1: Dashboard
        with gr.Tab("📊 Dashboard & Analytics"):
            with gr.Row():
                load_btn = gr.Button("🔄 Load Latest Experiment Data", variant="primary", size="lg")

            status_text = gr.Markdown("Click 'Load Latest Experiment Data' to begin")

            with gr.Row(visible=False) as metrics_row:
                metrics_plot = gr.Plot(label="Performance Metrics")

            with gr.Row(visible=False) as viz_row:
                with gr.Column():
                    class_plot = gr.Plot(label="Class Distribution")
                    amount_plot = gr.Plot(label="Amount Distribution")

                with gr.Column():
                    pca_plot = gr.Plot(label="PCA Visualization")
                    corr_plot = gr.Plot(label="Feature Correlation")

        # Tab 2: Single Prediction
        with gr.Tab("🔍 Single Transaction Prediction"):
            gr.Markdown("### Enter transaction details for fraud detection")

            data, _ = get_latest_run_data()
            if data is not None:
                feature_names = data['X_train'].columns.tolist()

                with gr.Row():
                    inputs = []
                    for i in range(0, len(feature_names), 3):
                        with gr.Column():
                            for feat in feature_names[i:i + 3]:
                                inp = gr.Number(label=feat, value=0.0)
                                inputs.append(inp)

                predict_btn = gr.Button("🎯 Predict", variant="primary", size="lg")

                with gr.Row():
                    with gr.Column():
                        prediction_result = gr.Markdown()
                    with gr.Column():
                        prediction_gauge = gr.Plot()

                predict_btn.click(
                    fn=predict_single_transaction,
                    inputs=inputs,
                    outputs=[prediction_result, prediction_gauge]
                )
            else:
                gr.Markdown("⚠️ Please load experiment data first from the Dashboard tab")

        # Tab 3: Batch Prediction
        with gr.Tab("📁 Batch Prediction"):
            gr.Markdown("### Upload a CSV file with transaction data for batch predictions")

            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            batch_predict_btn = gr.Button("🚀 Run Batch Prediction", variant="primary", size="lg")

            batch_summary = gr.Markdown()

            with gr.Row():
                with gr.Column():
                    batch_results = gr.Dataframe(label="Predictions")
                with gr.Column():
                    batch_viz = gr.Plot(label="Result Distribution")

            batch_predict_btn.click(
                fn=predict_batch_transactions,
                inputs=[file_input],
                outputs=[batch_summary, batch_results, batch_viz]
            )

        # Tab 4: About
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About This Application

            This fraud detection system uses machine learning to identify potentially
                        fraudulent credit card transactions.

            ### Features:
            - 📊 **Real-time Analytics**: View training data statistics and model performance
            - 🔍 **Single Predictions**: Analyze individual transactions
            - 📁 **Batch Processing**: Process multiple transactions at once
            - 🔗 **MLflow Integration**: Automatic loading of latest experiment data

            ### Model Information:
            - **Algorithm**: Random Forest Classifier
            - **Framework**: scikit-learn
            - **Experiment Tracking**: MLflow
            - **API Endpoint**: `http://127.0.0.1:50002/fraud_detection/invocations`

            ### Requirements:
            1. MLflow server running on `http://127.0.0.1:5000`
            2. Model serving on port 50002
            3. Completed training run with logged artifacts

            ### Training Pipeline:
            The model is trained using a Dagster pipeline that:
            1. Ingests and preprocesses data
            2. Trains a Random Forest model
            3. Logs metrics and artifacts to MLflow
            4. Deploys model for serving

            ---
            **Developed by**: Emmanuel Olateju + Mohammed
            **Course**: AIMS October 2025
            """)

    # Load button handler
    load_btn.click(
        fn=load_visualizations,
        inputs=[],
        outputs=[status_text, metrics_plot, class_plot, corr_plot,
                amount_plot, pca_plot, metrics_row, viz_row]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
