import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_risk_pie(df_predictions):
    """
    Renders risk level distribution pie chart.
    """
    risk_counts = df_predictions['Risk_Level'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index.tolist(),
        values=risk_counts.values.tolist(),
        hole=0.6,
        marker=dict(colors=["#2DD4BF", "#FFA62B", "#FF4B4B"]),
        textinfo='percent+label',
        hoverinfo='label+value'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=240
    )
    return fig

def plot_risk_vs_tenure(df_predictions):
    """
    Renders box plot comparing tenure distribution across risk levels.
    """
    fig = px.box(
        df_predictions,
        x='Risk_Level',
        y='tenure',
        color='Risk_Level',
        color_discrete_map={"High": "#FF4B4B", "Medium": "#FFA62B", "Low": "#2DD4BF"},
        labels={"Risk_Level": "Risk Level", "tenure": "Tenure (Months)"}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    )
    return fig

def plot_confusion_matrix(cm):
    """
    Renders interactive confusion matrix heatmap.
    """
    z = cm
    x_labels = ['Predicted Retained', 'Predicted Churned']
    y_labels = ['Actual Retained', 'Actual Churned']
    
    hover = [
        [f"True Negative: {cm[0][0]}", f"False Positive: {cm[0][1]}"],
        [f"False Negative: {cm[1][0]}", f"True Positive: {cm[1][1]}"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[[0, '#1E2130'], [0.5, '#6C63FF'], [1, '#2DD4BF']],
        showscale=False,
        text=[[str(val) for val in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 18, "color": "#FAFAFA"},
        hoverinfo="text",
        hovertext=hover
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickfont=dict(color='#FAFAFA', size=12)),
        yaxis=dict(tickfont=dict(color='#FAFAFA', size=12)),
        margin=dict(t=30, b=30, l=30, r=30),
        height=320
    )
    return fig

def plot_roc_curve(fpr, tpr, auc_score):
    """
    Renders ROC curve plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.2)', dash='dash'),
        name='Random Classifier'
    ))
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        line=dict(color='#2DD4BF', width=3),
        name=f"XGBoost (AUC = {auc_score:.4f})"
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FAFAFA'),
        xaxis=dict(title="False Positive Rate", gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        yaxis=dict(title="True Positive Rate", gridcolor='rgba(255,255,255,0.05)', zeroline=False),
        margin=dict(t=30, b=30, l=30, r=30),
        height=320,
        showlegend=False
    )
    return fig
