from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot_2x2_histograms(df, numerical_columns):
    """
    Create a 2x2 grid of histograms for the specified numerical columns in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        numerical_columns (list): List of numerical column names to plot.
    
    Returns:
        fig: Plotly figure object containing the histograms.
    """
    # Create a 2x2 subplot layout
    fig = make_subplots(rows=2, cols=2, subplot_titles=numerical_columns)

    # Add histograms to each subplot
    for i, category in enumerate(numerical_columns):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(go.Histogram(x=df[category], name=category, 
                                    marker=dict(color='#4E2A84', opacity=0.7)
                                   ), row=row, col=col)

    return fig