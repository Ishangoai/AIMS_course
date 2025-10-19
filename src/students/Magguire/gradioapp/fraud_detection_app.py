"""
Fraud Detection App with Tabbed Interface
Run locally with: python app_tabbed.py
"""
import base64
import io
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gradio.themes import Soft
from gradioapp.utils.fraud_detection import model

# Store transaction history for graphing
transaction_history = []


def create_evolution_graph():
    """Create a beautiful time evolution graph of fraud predictions"""
    if len(transaction_history) == 0:
        # Return a placeholder message when no data
        return """
        <div style="display: flex; align-items: center; justify-content: center; height: 400px;
                    background: linear-gradient(135deg, #252938 0%, #1a1d2e 100%);
                    border-radius: 10px; color: white; font-size: 18px; text-align: center; padding: 20px;">
            <div>
                <div style="font-size: 64px; margin-bottom: 20px;">📊</div>
                <div style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">No Data Yet</div>
                <div style="color: #94a3b8; font-size: 14px;">
                    Upload a file or analyze transactions to see the evolution graph
                </div>
            </div>
        </div>
        """

    # Create figure with dark theme - DOUBLED HEIGHT
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), facecolor='#1a1d2e')

    # Extract data
    times = [t['timestamp'] for t in transaction_history]
    fraud_probs = [t['fraud_prob'] for t in transaction_history]
    amounts = [t['amount'] for t in transaction_history]
    risk_levels = [t['risk_level'] for t in transaction_history]

    # Updated color mapping - more professional palette
    color_map = {
        'CRITICAL RISK': '#dc2626',
        'HIGH RISK': '#f97316',
        'MEDIUM RISK': '#eab308',
        'LOW RISK': '#22c55e',
        'MINIMAL RISK': '#10b981'
    }
    colors = [color_map[r] for r in risk_levels]

    # Plot 1: Fraud Probability Over Time
    ax1.set_facecolor('#252938')
    ax1.plot(times, fraud_probs, color='#6366f1', linewidth=3, marker='o',
             markersize=10, markerfacecolor='#8b5cf6', markeredgecolor='white',
             markeredgewidth=2.5, label='Fraud Probability', zorder=3)

    # Add horizontal lines for risk thresholds
    ax1.axhline(y=80, color='#dc2626', linestyle='--', linewidth=2.5, alpha=0.6, label='Critical (80%)')
    ax1.axhline(y=60, color='#f97316', linestyle='--', linewidth=2.5, alpha=0.6, label='High (60%)')
    ax1.axhline(y=40, color='#eab308', linestyle='--', linewidth=2.5, alpha=0.6, label='Medium (40%)')
    ax1.axhline(y=20, color='#22c55e', linestyle='--', linewidth=2.5, alpha=0.6, label='Low (20%)')

    # Fill areas between thresholds
    ax1.fill_between(times, 80, 100, color='#dc2626', alpha=0.15)
    ax1.fill_between(times, 60, 80, color='#f97316', alpha=0.15)
    ax1.fill_between(times, 40, 60, color='#eab308', alpha=0.15)
    ax1.fill_between(times, 20, 40, color='#22c55e', alpha=0.15)
    ax1.fill_between(times, 0, 20, color='#10b981', alpha=0.15)

    ax1.set_xlabel('Transaction Number', fontsize=14, color='white', fontweight='bold')
    ax1.set_ylabel('Fraud Probability (%)', fontsize=16, color='white', fontweight='bold')
    ax1.set_title('🔍 Fraud Detection Evolution Over Time', fontsize=18,
                  color='white', fontweight='bold', pad=25)
    ax1.legend(loc='upper right', framealpha=0.95, facecolor='#252938',
               edgecolor='white', fontsize=11)
    ax1.grid(True, alpha=0.25, color='white', linestyle=':', linewidth=1.2)
    ax1.tick_params(colors='white', labelsize=11)
    ax1.set_ylim(-5, 105)

    # Plot 2: Transaction Amounts with Risk Coloring
    ax2.set_facecolor('#252938')
    ax2.scatter(times, amounts, c=colors, s=250, alpha=0.85,
                edgecolors='white', linewidth=2.5, zorder=3)
    ax2.plot(times, amounts, color='#6366f1', linewidth=2.5, alpha=0.5, zorder=2)

    ax2.set_xlabel('Transaction Number', fontsize=14, color='white', fontweight='bold')
    ax2.set_ylabel('Transaction Amount ($)', fontsize=16, color='white', fontweight='bold')
    ax2.set_title('💰 Transaction Amounts by Risk Level', fontsize=18,
                  color='white', fontweight='bold', pad=25)
    ax2.grid(True, alpha=0.25, color='white', linestyle=':', linewidth=1.2)
    ax2.tick_params(colors='white', labelsize=11)

    # Add custom legend for risk levels
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#dc2626', label='🔴 Critical'),
        Patch(facecolor='#f97316', label='🟠 High'),
        Patch(facecolor='#eab308', label='🟡 Medium'),
        Patch(facecolor='#22c55e', label='🟢 Low'),
        Patch(facecolor='#10b981', label='✅ Minimal')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.95,
               facecolor='#252938', edgecolor='white', fontsize=11)

    plt.tight_layout()

    # Convert to base64 for display
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, facecolor='#1a1d2e', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return (f'<img src="data:image/png;base64,{img_base64}" style="width:100%; border-radius:10px;'
    'box-shadow: 0 4px 6px rgba(0,0,0,0.3);">')


def process_batch_file(file_path, progress=gr.Progress()):
    """Process Excel or CSV file with batch predictions"""
    if file_path is None:
        # Prompt user to upload a file
        prompt_html = """
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
                    border: 3px solid #f59e0b; padding: 25px;
                    border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center;">
            <div style="font-size: 48px; margin-bottom: 15px;">📁</div>
            <h3 style="color: #92400e; margin-top: 0; margin-bottom: 10px;">No File Uploaded</h3>
            <p style="color: #78350f; font-size: 15px; line-height: 1.6;">
                Please upload a CSV or Excel file to begin batch analysis.<br>
                Use the file upload area on the left to select your file.
            </p>
            <div style="margin-top: 15px; padding: 12px; background: white; border-radius: 8px;">
                <p style="color: #92400e; font-size: 13px; margin: 0; font-weight: 600;">
                    💡 Tip: Your file should contain columns for V14, V10, V17, V16, V3, V12, V4, V18, V11,
                    Time, and Amount
                </p>
            </div>
        </div>
        """
        return prompt_html, create_evolution_graph()

    try:
        progress(0, desc="📂 Reading file...")

        # Read file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        progress(0.1, desc="✅ File loaded successfully")

        # Expected columns for the simplified model
        required_cols = ['V14', 'V10', 'V17', 'V16', 'V3', 'V12', 'V4', 'V18', 'V11', 'Time', 'Amount']

        # Check if columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_html = f"""
            <div style="background: #fef2f2; border: 3px solid #dc2626; padding: 25px;
                        border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h3 style="color: #dc2626; margin-top: 0;">❌ Missing Required Columns</h3>
                <p style="color: #64748b; font-size: 15px;">The following required
                columns are missing from your file:</p>
                <div style="background: #fee; padding: 12px; border-radius: 8px; margin: 10px 0;">
                    <p style="color: #dc2626; font-weight: bold; margin: 0;">{', '.join(missing_cols)}</p>
                </div>
                <p style="color: #64748b; font-size: 13px; margin-top: 15px;">
                    Please ensure your file includes all required columns: V14,
                    V10, V17, V16, V3, V12, V4, V18, V11, Time, Amount
                </p>
            </div>
            """
            return error_html, create_evolution_graph()

        progress(0.2, desc="🔍 Validating data structure...")

        # Clear previous history
        transaction_history.clear()

        total_rows = len(df)
        predictions = []

        # Make predictions for all rows
        for idx, row in df.iterrows():
            # Update progress
            progress_value = 0.2 + (0.7 * (idx + 1) / total_rows)  # type: ignore
            progress(progress_value, desc=f"🔄 Processing transaction {idx + 1}/{total_rows}...")  # type: ignore

            # Prepare features in the correct order
            features = np.array([[
                row['V14'], row['V10'], row['V17'], row['V16'], row['V3'],
                row['V12'], row['V4'], row['V18'], row['V11'],
                row['Time'], row['Amount']
            ]])

            # Predict
            probability = model.predict_proba(features)[0]
            fraud_prob = probability[1] * 100

            # Determine risk level
            if fraud_prob >= 80:
                risk_level = "CRITICAL RISK"
            elif fraud_prob >= 60:
                risk_level = "HIGH RISK"
            elif fraud_prob >= 40:
                risk_level = "MEDIUM RISK"
            elif fraud_prob >= 20:
                risk_level = "LOW RISK"
            else:
                risk_level = "MINIMAL RISK"

            # Store prediction
            predictions.append({
                'fraud_prob': fraud_prob,
                'risk_level': risk_level
            })

            # Add to history for graphing
            transaction_history.append({
                'timestamp': idx + 1,  # type: ignore
                'fraud_prob': fraud_prob,
                'amount': row['Amount'],
                'risk_level': risk_level,
                'time': row['Time']
            })

        progress(0.9, desc="📊 Generating statistics...")

        # Calculate statistics
        fraud_count = sum(1 for p in predictions if p['fraud_prob'] > 50)
        legitimate_count = len(predictions) - fraud_count

        critical_count = sum(1 for p in predictions if p['risk_level'] == 'CRITICAL RISK')
        high_count = sum(1 for p in predictions if p['risk_level'] == 'HIGH RISK')
        medium_count = sum(1 for p in predictions if p['risk_level'] == 'MEDIUM RISK')
        low_count = sum(1 for p in predictions if p['risk_level'] == 'LOW RISK')
        minimal_count = sum(1 for p in predictions if p['risk_level'] == 'MINIMAL RISK')

        avg_fraud_prob = np.mean([p['fraud_prob'] for p in predictions])
        max_amount = df['Amount'].max()
        min_amount = df['Amount'].min()
        avg_amount = df['Amount'].mean()

        progress(1.0, desc="✅ Analysis complete!")

        # Create statistics HTML with updated colors
        stats_html = f"""
        <div style="font-family: Arial, sans-serif; padding:12px; border-radius:12px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            box-shadow:0 6px 18px rgba(0,0,0,0.2);">

    <h2 style="color:white; text-align:center; margin:8px 0 12px; font-size:22px;">
        📊 Batch Analysis Statistics
    </h2>

    <div style="background:white; padding:12px; border-radius:10px; margin-bottom:12px;
                box-shadow:0 2px 6px rgba(0,0,0,0.1);">
        <h3 style="color:#333; margin:0 0 6px; font-size:16px;">📈 Overview</h3>
        <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:10px;">
            <div style="background:#f8fafc; padding:10px; border-radius:6px; border-left:4px solid #6366f1;">
                <p style="margin:0; color:#64748b; font-size:12px;">Total Transactions</p>
                <p style="margin:3px 0 0; color:#1e293b; font-size:20px; font-weight:bold;">{len(predictions)}</p>
            </div>
            <div style="background:#fef2f2; padding:10px; border-radius:6px; border-left:4px solid #dc2626;">
                <p style="margin:0; color:#64748b; font-size:12px;">🚨 Fraud Detected</p>
                <p style="margin:3px 0 0; color:#dc2626; font-size:20px; font-weight:bold;">
                    {fraud_count} ({fraud_count / len(predictions) * 100:.1f}%)
                </p>
            </div>
            <div style="background:#f0fdf4; padding:10px; border-radius:6px; border-left:4px solid #16a34a;">
                <p style="margin:0; color:#64748b; font-size:12px;">✅ Legitimate</p>
                <p style="margin:3px 0 0; color:#16a34a; font-size:20px; font-weight:bold;">
                    {legitimate_count} ({legitimate_count / len(predictions) * 100:.1f}%)
                </p>
            </div>
            <div style="background:#fef9f0; padding:10px; border-radius:6px; border-left:4px solid #f97316;">
                <p style="margin:0; color:#64748b; font-size:12px;">📊 Avg Fraud Probability</p>
                <p style="margin:3px 0 0; color:#f97316; font-size:20px; font-weight:bold;">
                    {avg_fraud_prob:.1f}%
                </p>
            </div>
        </div>
    </div>

    <div style="background:white; padding:12px; border-radius:10px; margin-bottom:12px;
                box-shadow:0 2px 6px rgba(0,0,0,0.1);">
        <h3 style="color:#333; margin:0 0 6px; font-size:16px;">🎯 Risk Level Distribution</h3>
        <div style="display:grid; grid-template-columns:repeat(5,1fr); gap:6px;">
            <div style="text-align:center; padding:8px; background:#fef2f2; border-radius:6px;">
                <div style="font-size:26px; margin-bottom:3px;">🔴</div>
                <p style="margin:0; color:#dc2626; font-weight:bold; font-size:18px;">{critical_count}</p>
                <p style="margin:3px 0 0; color:#64748b; font-size:11px;">Critical</p>
            </div>
            <div style="text-align:center; padding:8px; background:#fff7ed; border-radius:6px;">
                <div style="font-size:26px; margin-bottom:3px;">🟠</div>
                <p style="margin:0; color:#f97316; font-weight:bold; font-size:18px;">{high_count}</p>
                <p style="margin:3px 0 0; color:#64748b; font-size:11px;">High</p>
            </div>
            <div style="text-align:center; padding:8px; background:#fefce8; border-radius:6px;">
                <div style="font-size:26px; margin-bottom:3px;">🟡</div>
                <p style="margin:0; color:#eab308; font-weight:bold; font-size:18px;">{medium_count}</p>
                <p style="margin:3px 0 0; color:#64748b; font-size:11px;">Medium</p>
            </div>
            <div style="text-align:center; padding:8px; background:#f0fdf4; border-radius:6px;">
                <div style="font-size:26px; margin-bottom:3px;">🟢</div>
                <p style="margin:0; color:#22c55e; font-weight:bold; font-size:18px;">{low_count}</p>
                <p style="margin:3px 0 0; color:#64748b; font-size:11px;">Low</p>
            </div>
            <div style="text-align:center; padding:8px; background:#ecfdf5; border-radius:6px;">
                <div style="font-size:26px; margin-bottom:3px;">✅</div>
                <p style="margin:0; color:#10b981; font-weight:bold; font-size:18px;">{minimal_count}</p>
                <p style="margin:3px 0 0; color:#64748b; font-size:11px;">Minimal</p>
            </div>
        </div>
    </div>

    <div style="background:white; padding:12px; border-radius:10px; box-shadow:0 2px 6px rgba(0,0,0,0.1);">
        <h3 style="color:#333; margin:0 0 6px; font-size:16px;">💰 Transaction Details</h3>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:10px;">
            <div style="background:#f8fafc; padding:10px; border-radius:6px;">
                <p style="margin:0; color:#64748b; font-size:12px;">💵 Max Amount</p>
                <p style="margin:3px 0 0; color:#1e293b; font-size:18px; font-weight:bold;">${max_amount:.2f}</p>
            </div>
            <div style="background:#f8fafc; padding:10px; border-radius:6px;">
                <p style="margin:0; color:#64748b; font-size:12px;">💵 Min Amount</p>
                <p style="margin:3px 0 0; color:#1e293b; font-size:18px; font-weight:bold;">${min_amount:.2f}</p>
            </div>
            <div style="background:#f8fafc; padding:10px; border-radius:6px;">
                <p style="margin:0; color:#64748b; font-size:12px;">💵 Avg Amount</p>
                <p style="margin:3px 0 0; color:#1e293b; font-size:18px; font-weight:bold;">${avg_amount:.2f}</p>
            </div>
        </div>
    </div>
</div>
"""

        # Generate graph
        graph_html = create_evolution_graph()

        return stats_html, graph_html

    except Exception as e:
        error_html = f"""
        <div style="background: #fef2f2; border: 3px solid #dc2626; padding: 25px;
                    border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #dc2626; margin-top: 0;">❌ File Processing Error</h3>
            <p style="color: #64748b; font-size: 15px;">An error occurred while processing the file:</p>
            <pre style="background: #f8fafc; padding: 15px; border-radius: 8px; overflow-x: auto; color: #1e293b;">
{str(e)}</pre>
        </div>
        """
        return error_html, create_evolution_graph()


def predict_fraud(V14, V10, V17, V16, V3, V12, V4, V18, V11, time, amount):
    """Predict fraud and return styled results with evolution graph"""
    try:
        # Prepare features
        features = np.array([[V14, V10, V17, V16, V3, V12, V4, V18, V11, time, amount]])

        # Get prediction
        probability = model.predict_proba(features)[0]

        fraud_prob = probability[1] * 100
        legitimate_prob = probability[0] * 100

        # Determine risk level with updated colors
        if fraud_prob >= 80:
            risk_emoji = "🔴"
            risk_level = "CRITICAL RISK"
            risk_color = "#dc2626"
            alert = "⚠️ HIGH FRAUD ALERT - BLOCK TRANSACTION"
            recommendation = "🚫 Block immediately and flag account for investigation."
        elif fraud_prob >= 60:
            risk_emoji = "🟠"
            risk_level = "HIGH RISK"
            risk_color = "#f97316"
            alert = "⚠️ FRAUD WARNING - MANUAL REVIEW REQUIRED"
            recommendation = "⚡ Require additional verification before processing."
        elif fraud_prob >= 40:
            risk_emoji = "🟡"
            risk_level = "MEDIUM RISK"
            risk_color = "#eab308"
            alert = "⚡ MODERATE ALERT - MONITOR CLOSELY"
            recommendation = "👀 Apply enhanced monitoring and verification."
        elif fraud_prob >= 20:
            risk_emoji = "🟢"
            risk_level = "LOW RISK"
            risk_color = "#22c55e"
            alert = "✓ Low Risk - Standard Processing"
            recommendation = "✅ Process with standard security measures."
        else:
            risk_emoji = "✅"
            risk_level = "MINIMAL RISK"
            risk_color = "#10b981"
            alert = "✓ Transaction Appears Legitimate"
            recommendation = "✅ Safe to process normally."

        # Store in history (keep last 50 transactions)
        transaction_history.append({
            'timestamp': len(transaction_history) + 1,
            'fraud_prob': fraud_prob,
            'amount': amount,
            'risk_level': risk_level,
            'time': datetime.now()
        })
        if len(transaction_history) > 50:
            transaction_history.pop(0)

        # Create results HTML with updated colors
        result_html = f"""
        <div style="font-family: Arial, sans-serif; padding: 18px; border-radius: 12px;
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            box-shadow: 0 6px 18px rgba(0,0,0,0.18);">

    <h2 style="color: white; text-align: center; margin: 10px 0 18px; font-size: 20px;">
        🛡️ Fraud Detection — Summary
    </h2>

    <div style="display: flex; gap: 16px; flex-wrap: wrap;">
        <div style="flex: 1 1 calc(50% - 8px); min-width: 260px;
                    background: white; padding: 16px; border-radius: 10px; margin-bottom: 16px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h3 style="color:#1e293b; margin:0 0 10px; font-size:16px;">🎯 Risk</h3>
            <div style="background: {risk_color}; color: white; padding:12px; border-radius:8px;
                        text-align:center; font-size:14px; font-weight:700; margin-bottom:12px;">
                {risk_emoji} {risk_level}
            </div>
            <div style="background:#f8fafc; padding:12px; border-radius:8px; border-left:4px solid {risk_color};">
                <p style="margin:0; font-size:13px; color:#1e293b; font-weight:600;">{alert}</p>
            </div>
        </div>

        <div style="flex: 1 1 calc(50% - 8px); min-width:260px;
                    background:white; padding:16px; border-radius:10px; margin-bottom:16px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h3 style="color:#1e293b; margin:0 0 10px; font-size:16px;">📊 Confidence</h3>
            <div style="margin-bottom:12px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:#dc2626; font-weight:700; font-size:13px;">🚨 Fraud</span>
                    <span style="color:#dc2626; font-weight:700; font-size:14px;">{fraud_prob:.1f}%</span>
                </div>
                <div style="background:#e5e7eb; border-radius:12px; overflow:hidden; height:26px;">
                    <div style="background:linear-gradient(90deg,#f87171,#dc2626); width:{fraud_prob}%;
                                height:100%; display:flex; align-items:center; justify-content:center;
                                color:white; font-weight:700; font-size:12px;">
                        {fraud_prob:.1f}%
                    </div>
                </div>
            </div>
            <div>
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:#16a34a; font-weight:700; font-size:13px;">✅ Legit</span>
                    <span style="color:#16a34a; font-weight:700; font-size:14px;">{legitimate_prob:.1f}%</span>
                </div>
                <div style="background:#e5e7eb; border-radius:12px; overflow:hidden; height:26px;">
                    <div style="background:linear-gradient(90deg,#86efac,#16a34a); width:{legitimate_prob}%;
                                height:100%; display:flex; align-items:center; justify-content:center;
                                color:white; font-weight:700; font-size:12px;">
                        {legitimate_prob:.1f}%
                    </div>
                </div>
            </div>
        </div>

        <div style="flex: 1 1 100%; background:white; padding:16px; border-radius:10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.08);">
            <h3 style="color:#1e293b; margin:0 0 10px; font-size:16px;">💡 Recommendation</h3>
            <p style="color:#475569; font-size:13px; line-height:1.5; margin:8px 0;">
                {recommendation}
            </p>
            <div style="margin-top:12px; padding:12px; background:#f8fafc;
            border-radius:8px; border-left:4px solid #6366f1;">
                <p style="margin:4px 0; color:#64748b; font-size:13px;">
                    <strong>💰 Amount:</strong> ${amount:.2f}
                </p>
                <p style="margin:4px 0; color:#64748b; font-size:13px;">
                    <strong>⏰ Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </p>
                <p style="margin:4px 0; color:#64748b; font-size:13px;">
                    <strong>🎯 Confidence:</strong> {max(fraud_prob, legitimate_prob):.1f}%
                </p>
                <p style="margin:4px 0; color:#64748b; font-size:13px;">
                    <strong>📋 Decision:</strong> {'REJECT ❌' if fraud_prob > 50 else 'APPROVE ✅'}
                </p>
                <p style="margin:4px 0; color:#64748b; font-size:13px;">
                    <strong>📊 Transactions Analysed:</strong> {len(transaction_history)}
                </p>
            </div>
        </div>
    </div>
</div>

        """

        # Generate evolution graph
        graph_html = create_evolution_graph()

        return result_html, graph_html

    except Exception as e:
        error_html = f"""
        <div style="background: #fef2f2; border: 3px solid #dc2626; padding: 25px;
                    border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #dc2626; margin-top: 0;">❌ Prediction Error</h3>
            <p style="color: #64748b; font-size: 15px;">An error occurred during prediction:</p>
            <pre style="background: #f8fafc; padding: 15px; border-radius: 8px; overflow-x: auto; color: #1e293b;">
{str(e)}</pre>
        </div>
        """
        return error_html, create_evolution_graph()


def clear_history():
    """Clear transaction history"""
    transaction_history.clear()
    return "✅ Transaction history cleared!", create_evolution_graph()


# Custom CSS with updated colors and tab text visibility
custom_css = """
#excel_upload {
    height: 150px;
}
.gradio-container {
    font-family: 'Segoe UI', Arial, sans-serif !important;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%) !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    font-weight: bold !important;
    font-size: 18px !important;
    padding: 15px 30px !important;
    border-radius: 10px !important;
    box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s !important;
}
.gr-button-primary:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 20px rgba(99, 102, 241, 0.6) !important;
}
.gr-button-secondary {
    background: linear-gradient(135deg, #ec4899 0%, #f43f5e 100%) !important;
    border: none !important;
    font-weight: bold !important;
    color: white !important;
}

/* Fix tab text visibility - ensure black text on light background */
.tab-nav button {
    color: #1e293b !important;
    font-weight: 600 !important;
    background: #f8fafc !important;
    border: 2px solid #e2e8f0 !important;
}
.tab-nav button[aria-selected="true"] {
    color: #1e293b !important;
    background: white !important;
    border-bottom: 3px solid #6366f1 !important;
}
.tab-nav button:hover {
    color: #0f172a !important;
    background: #e2e8f0 !important;
}

/* Ensure all text in tabs is visible */
.tabs {
    color: #1e293b !important;
}
"""

# Create Gradio Interface with Tabs
with gr.Blocks(css=custom_css, theme=Soft(), title="Fraud Detection System") as fraud_app:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🛡️ Credit Card Fraud Detection System</h1>
            <h3>Powered by Machine Learning - Real-time Transaction Analysis with Evolution Tracking</h3>
        </div>
        """
    )

    with gr.Tabs():
        # TAB 1: Manual Input
        with gr.Tab("📝 Single Transaction Analysis"):
            gr.Markdown(
                """
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
                     padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #6366f1;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b;">📖 How to Use This Tab</h3>
                    <p style="margin: 0; color: #475569; line-height: 1.6;">
                        <strong>1.</strong> Enter the PCA feature values (V14, V10, V17, etc.) in the input fields<br>
                        <strong>2.</strong> Provide the transaction Time (in seconds) and Amount (in dollars)<br>
                        <strong>3.</strong> Click <strong>"Analyze Transaction"</strong> to get instant fraud risk
                        assessment<br>
                        <strong>4.</strong> View results including risk level, confidence scores,
                         and recommendations<br>
                        <strong>5.</strong> Track transaction history in the evolution graph on the right
                    </p>
                </div>
                """
            )

            with gr.Row():
                # LEFT COLUMN - Input Form
                with gr.Column(scale=1):
                    gr.Markdown("### 🔢 PCA Features")
                    with gr.Accordion("📊 Main Features", open=True):
                        with gr.Row():
                            v14 = gr.Number(label="V14", value=0.35237478)
                            v10 = gr.Number(label="V10", value=-0.156840906)
                            v17 = gr.Number(label="V17", value=-80.010299228)

                    with gr.Accordion("📊 Secondary features", open=False):
                        with gr.Row():
                            v16 = gr.Number(label="V16", value=0.906489408)
                            v3 = gr.Number(label="V3", value=0.220932254)
                            v12 = gr.Number(label="V12", value=0.538299087)
                        with gr.Row():
                            v4 = gr.Number(label="V4", value=-0.146817811)
                            v18 = gr.Number(label="V18", value=740.576837916)
                            v11 = gr.Number(label="V11", value=0.847755891)

                    with gr.Row():
                        time_input = gr.Number(label="⏰ Time (seconds)", value=62800.0)
                        amount_input = gr.Number(label="💰 Transaction Amount ($)", value=8.99)

                    with gr.Row():
                        predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
                        clear_btn_manual = gr.Button("🗑️ Clear History", variant="secondary", size="lg")

                # MIDDLE COLUMN - Results
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="text-align: center;">
                        <h3>📈 Analysis Results</h3>
                    </div>"""
                    )
                    output_manual = gr.HTML(label="Prediction Results", value="")

                # RIGHT COLUMN - Evolution Graph
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="text-align: center;">
                        <h3>📊 Evolution Over Time</h3>
                    </div>
                    """)
                    graph_output_manual = gr.HTML(
                        label="Transaction Evolution Graph",
                        value=create_evolution_graph()  # Initialize with placeholder
                    )

            clear_status_manual = gr.Textbox(label="Status", visible=False)

            with gr.Row():
                gr.Markdown(
                    """
                    ---
                    ### 💡 Tips:
                    - **PCA Features (V14-V18)**: These are transformed features from Principal Component Analysis
                    - **Time**: Represents seconds elapsed since first transaction in dataset
                    - **Amount**: Transaction amount in dollars
                    - **Risk Levels**: Critical (>80%), High (60-80%), Medium (40-60%), Low (20-40%), Minimal (<20%)
                    """
                )

        # TAB 2: File Upload
        with gr.Tab("📂 Batch File Analysis"):
            gr.Markdown(
                """
                <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
                     padding: 15px; border-radius: 10px; margin-bottom: 20px; border-left: 4px solid #6366f1;">
                    <h3 style="margin: 0 0 8px 0; color: #1e293b;">📖 How to Use This Tab</h3>
                    <p style="margin: 0; color: #475569; line-height: 1.6;">
                        <strong>1.</strong> Prepare a CSV or XLSX file with the required columns (see format below)<br>
                        <strong>2.</strong> Click the upload area and select your file<br>
                        <strong>3.</strong> Click <strong>"Analyze Batch File"</strong> to process all transactions<br>
                        <strong>4.</strong> View comprehensive statistics including fraud counts
                         and risk distribution<br>
                        <strong>5.</strong> Examine the evolution graph showing all transactions' risk levels
                    </p>
                </div>
                """
            )

            with gr.Row():
                # LEFT COLUMN - File Upload
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("#### 📂 Upload Excel/CSV File")
                        excel_file = gr.File(
                            label="Upload file with columns: V14, V10, V17, V16, V3, V12, V4, V18, V11, Time, Amount",
                            file_types=[".xlsx", ".csv"],
                            type="filepath",
                            elem_id="excel_upload"
                        )
                        process_file_btn = gr.Button("📊 Analyze Batch File", variant="primary", size="lg")
                        clear_btn_batch = gr.Button("🗑️ Clear History", variant="secondary", size="lg")

                # MIDDLE COLUMN - Results/Statistics
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="text-align: center;">
                        <h3>📈 Analysis Statistics</h3>
                    </div>"""
                    )
                    output_batch = gr.HTML(label="Batch Results", value="")

                # RIGHT COLUMN - Evolution Graph
                with gr.Column(scale=1):
                    gr.Markdown("""
                    <div style="text-align: center;">
                        <h3>📊 Evolution Over Time</h3>
                    </div>
                    """)
                    graph_output_batch = gr.HTML(
                        label="Transaction Evolution Graph",
                        value=create_evolution_graph()  # Initialize with placeholder
                    )

            clear_status_batch = gr.Textbox(label="Status", visible=False)

            with gr.Row():
                gr.Markdown(
                    """
                    ---
                    ### 📄 File Format Example:
                    ```csv
V14,V10,V17,V16,V3,V12,V4,V18,V11,Time,Amount
0.35,-0.16,-80.01,0.91,0.22,0.54,-0.15,740.58,0.85,62800,8.99
-1.23,0.45,-65.32,1.20,0.88,0.33,-0.22,450.21,0.92,63000,150.50
0.78,-0.89,-45.67,0.65,1.10,0.75,-0.33,320.45,0.70,63200,75.25
```

                    ### 📋 Required Columns:
                    - **V14, V10, V17, V16, V3, V12, V4, V18, V11**: PCA-transformed features
                    - **Time**: Transaction time in seconds
                    - **Amount**: Transaction amount in dollars

                    ### 💾 Supported Formats:
                    - **CSV** (.csv) - Comma-separated values
                    - **Excel** (.xlsx) - Microsoft Excel format
                    """
                )

    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
             border-radius: 10px; margin-top: 20px;">
            <h3 style="color: #1e293b; margin-bottom: 15px;">📊 Model Information</h3>
            <div style="display: flex; justify-content: center; gap: 30px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <p style="color: #6366f1; font-weight: bold; font-size: 18px; margin: 5px 0;">🤖 Algorithm</p>
                    <p style="color: #64748b; margin: 0;">Random Forest Classifier</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: #8b5cf6; font-weight: bold; font-size: 18px; margin: 5px 0;">🎯 Features</p>
                    <p style="color: #64748b; margin: 0;">11 Key Components</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: #ec4899; font-weight: bold; font-size: 18px; margin: 5px 0;">⚡ Performance</p>
                    <p style="color: #64748b; margin: 0;">Real-time Detection</p>
                </div>
                <div style="text-align: center;">
                    <p style="color: #10b981; font-weight: bold; font-size: 18px; margin: 5px 0;">🔍 Optimization</p>
                    <p style="color: #64748b; margin: 0;">GridSearchCV</p>
                </div>
            </div>
            <p style="color: #94a3b8; font-size: 12px; margin-top: 15px;">
                © 2025 Fraud Detection System | Built with Gradio & Machine Learning
            </p>
        </div>
        """
    )

    # Connect buttons for Manual Input tab
    predict_btn.click(
        fn=predict_fraud,
        inputs=[v14, v10, v17, v16, v3, v12, v4, v18, v11, time_input, amount_input],
        outputs=[output_manual, graph_output_manual]
    )

    clear_btn_manual.click(
        fn=clear_history,
        inputs=[],
        outputs=[clear_status_manual, graph_output_manual]
    )

    # Connect buttons for Batch File tab
    process_file_btn.click(
        fn=process_batch_file,
        inputs=[excel_file],
        outputs=[output_batch, graph_output_batch]
    )

    clear_btn_batch.click(
        fn=clear_history,
        inputs=[],
        outputs=[clear_status_batch, graph_output_batch]
    )
