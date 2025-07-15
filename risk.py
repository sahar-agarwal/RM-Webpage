import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash 
from dash import dcc, html 
from dash.dependencies import Input, Output

# Generating random returns for some underlying asset
np.random.seed(42)
returns = np.random.normal(0, 0.02, 1000) # We are considering 1000 data points.
returns_df = pd.DataFrame(returns, columns=['Returns'])

# Calculating  Historical Value at Risk (VaR)
confidence = 0.95
ThresholdVaR = np.percentile(returns_df['Returns'], (1-confidence)*100)
print(f'VaR (hist.) given a confidence level of 95% = {ThresholdVaR: .4f}')

# Calculating Parametric Value at Risk (VaR)
mean_return = returns_df['Returns'].mean()
std_return = returns_df['Returns'].std()
# Considering a z-score of 1.645 for a 95% confidence level
z_score = 1.645
ParaVaR = mean_return-z_score*std_return
print(f'VaR (para.) given a confidence level of 95% (z-score = 1.645): {ParaVaR: .4f}')

# Generating portfolio values and calculating their VaRs
n_sims = 10000
sim_returns = np.random.normal(mean_return, std_return, n_sims)
MonteCarloVaR = np.percentile(sim_returns, (1-confidence)*100)
print(f'VaR (Monte Carlo) given a confidence level of 95%: {MonteCarloVaR: .4f}')

# Calculating the expected loss beyond VaR (CVaR)
cvar_vals = returns_df[returns_df['Returns'] <= ThresholdVaR]
HistCVaR = cvar_vals['Returns'].mean()
print(f'CVaR (hist.) given a 95% level of confidence: {HistCVaR: .4f}')

cvar_sim_vals = sim_returns[sim_returns <= MonteCarloVaR]
MonteCarloCVaR = np.mean(cvar_sim_vals)
print(f'CVaR (Monte Carlo) given a95% level of confidence: {MonteCarloCVaR: .4f}')

# Creating the Web-app

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H2('Risk Managing Dashboard (VaR & CVaR)'),
    html.Br(),
    html.Label('Confidence Level (%)'),
    html.Br(),
    dcc.Slider(id = 'confidence-level-slider', min = 80, max = 99, step = 1, value = 95,
               marks = {i: f'{i}%' for i in range(80, 100)}),
    html.Br(),
    html.Label('Select a Method to Calculate VaR'),
    html.Br(),
    dcc.Dropdown(
        id = 'var-method-selection',
        options = [
            {'label': 'Historical VaR', 'value': 'historical'},
            {'label': 'Parametric VaR', 'value': 'parametric'},
            {'label': 'Monte Carlo VaR', 'value': 'monte carlo'}
        ],
        value = 'historical'
    ),

    html.Div(id = 'var-output', style = {'padding': '20px', 'fontSize': '20px'}),
    dcc.Graph(id = 'return-histogram')
])

@app.callback(
    [Output('var-output', 'children'), Output('return-histogram', 'figure')],
    [Input('confidence-level-slider', 'value'), Input('var-method-selection', 'value')]
    )

def update_var(conf, method):
    confidence = conf/100 
    if method == 'historical':
        ThresholdVaR = np.percentile(returns_df['Returns'], (1-confidence)*100)
        cvar_val = returns_df[returns_df['Returns'] <= ThresholdVaR]['Returns'].mean()
    elif method == 'parametric':
        ThresholdVaR = mean_return-z_score*std_return
        cvar_val = returns_df[returns_df['Returns'] <= ThresholdVaR]['Returns'].mean()
    elif method == 'monte carlo':
        sim_returns = np.random.normal(mean_return, std_return, n_sims)
        ThresholdVaR = np.percentile(sim_returns, (1-confidence)*100)
        cvar_val = np.mean(sim_returns[sim_returns <= ThresholdVaR])

    var_text = f'{method.title()} VaR at {conf}% confidence: {ThresholdVaR: .4f}'
    cvar_text = f'{method.title()} CVaR at {conf}% confidence: {cvar_val: .4f}'

    fig = px.histogram(
        returns_df, 
        x = "Returns", 
        nbins = 250, 
        title = "Return Distribution", 
        color_discrete_sequence=['#A9A9A9'])
    
    fig.add_vline(x = ThresholdVaR, line = dict(color = 'red', width = 2), name = "VaR")
    fig.add_vline(x = cvar_val, line = dict(color ='blue', width = 2, dash = 'dot'), name = "CVaR")
    
    return[
        html.Div([
            html.Div(var_text),
            html.Br(),
            html.Div(cvar_text)
        ])
    ], fig

if __name__ == '__main__':
    app.run_server(debug=True)
