import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
from flask import Flask, render_template
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import datetime

# ----------------------------
# Flask Setup
# ----------------------------
server = Flask(__name__)


@server.route('/')
def index():
    return render_template('index.html')


# ----------------------------
# Data Loading & Preparation
# ----------------------------
df = pd.read_excel("DataAnalyst_Assesment_Dataset.xlsx")

# Data cleaning and formatting steps:
df.rename(columns={'Price': 'Total Revenue'}, inplace=True)
df['Booking Date'] = pd.to_datetime(df['Booking Date'], errors='coerce')
df['Booking Month'] = df['Booking Date'].dt.month
df['Time Slot'] = df['Time Slot'].astype(str)
df['Booking Hour'] = df['Time Slot'].str.extract(r'(\d{1,2})').astype(float)

# Create a 'Booking Type' feature based on keywords in 'Service Name'
df['Booking Type'] = (
    df['Service Name'].str.contains('Subscription').map({True: 'Subscription', False: ''}) +
    df['Service Name'].str.contains('Party').map({True: 'Birthday Party', False: ''}) +
    df['Service Name'].str.contains('Rental').map({True: 'Facility Rental', False: ''})
)
df.loc[df['Booking Type'] == '', 'Booking Type'] = 'Class Booking'

# Aggregate data by date for forecasting
daily_revenue = df.groupby('Booking Date')['Total Revenue'].sum().reset_index()
daily_revenue = daily_revenue.sort_values('Booking Date')

# ----------------------------
# Pre-Generate Figures for Static Tabs
# ----------------------------
# 1. Total Revenue Per Month (Line Chart)
fig_revenue_month = px.line(
    df.groupby('Booking Month')['Total Revenue'].sum().reset_index(),
    x='Booking Month', y='Total Revenue',
    title="Total Revenue Per Month",
    labels={'Booking Month': 'Month', 'Total Revenue': 'Revenue ($)'}
)

# 2. Revenue by Service Name (Bar Chart)
fig_revenue_service = px.bar(
    df.groupby('Service Name')['Total Revenue'].sum().reset_index().sort_values(by='Total Revenue', ascending=False),
    x='Service Name', y='Total Revenue',
    title="Revenue by Service Name",
    color='Total Revenue'
)

# 3. Distribution of Booking Types (Pie Chart)
fig_booking_type = px.pie(
    df,
    names='Booking Type',
    title="Distribution of Booking Types"
)

# 4. Peak Booking Hours (Histogram)
fig_booking_hour = px.histogram(
    df,
    x='Booking Hour',
    nbins=24,
    title="Peak Booking Hours",
    labels={'Booking Hour': 'Hour of Day', 'count': 'Number of Bookings'}
)

# ----------------------------
# Dash App Setup with Multiple Tabs (including Forecast)
# ----------------------------
dash_app = dash.Dash(
    __name__,
    server=server,
    url_base_pathname='/dashboard/',
    suppress_callback_exceptions=True
)

dash_app.layout = html.Div(children=[
    html.H1("Booking Revenue Dashboard", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label='Revenue Overview', children=[
            html.Div([
                html.H2("Total Revenue Per Month", style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_revenue_month)
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Service Revenue', children=[
            html.Div([
                html.H2("Revenue by Service Name", style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_revenue_service)
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Booking Analysis', children=[
            html.Div([
                html.H2("Distribution of Booking Types", style={'textAlign': 'center'}),
                dcc.Graph(figure=fig_booking_type),
                html.H2("Peak Booking Hours", style={'textAlign': 'center', 'marginTop': '40px'}),
                dcc.Graph(figure=fig_booking_hour)
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Interactive Booking Trend', children=[
            html.Div([
                html.H2("Interactive Revenue Trend by Booking Type", style={'textAlign': 'center'}),
                html.Div([
                    html.Label("Select Booking Type:"),
                    dcc.Dropdown(
                        id='booking-type-dropdown',
                        options=[{'label': btype, 'value': btype} for btype in df['Booking Type'].unique()],
                        value=df['Booking Type'].unique()[0],
                        clearable=False,
                        style={'width': '50%', 'margin': 'auto'}
                    )
                ], style={'padding': '20px'}),
                dcc.Graph(id='revenue-trend-chart')
            ], style={'padding': '20px'})
        ]),
        dcc.Tab(label='Forecast Revenue', children=[
            html.Div([
                html.H2("Forecast Future Revenue", style={'textAlign': 'center'}),
                html.Div([
                    # Forecast horizon (days into the future)
                    html.Label("Enter Forecast Horizon (Days):"),
                    dcc.Input(
                        id='forecast-days',
                        type='number',
                        value=30,
                        min=1,
                        style={'marginRight': '10px'}
                    ),
                    # Training window (days of past data to use)
                    html.Label("Training Window (Days):"),
                    dcc.Input(
                        id='training-days',
                        type='number',
                        value=90,
                        min=1,
                        style={'marginRight': '10px'}
                    ),
                    # Model type dropdown
                    html.Label("Select Forecast Model:"),
                    dcc.Dropdown(
                        id='forecast-model-dropdown',
                        options=[
                            {'label': 'Linear Regression', 'value': 'linear'},
                            {'label': 'Polynomial (degree=2)', 'value': 'poly2'},
                            {'label': 'Polynomial (degree=3)', 'value': 'poly3'}
                        ],
                        value='linear',
                        clearable=False,
                        style={'width': '40%', 'margin': 'auto'}
                    ),
                    html.Button('Generate Forecast', id='forecast-button', n_clicks=0)
                ], style={'textAlign': 'center', 'padding': '20px'}),
                dcc.Graph(id='forecast-graph')
            ], style={'padding': '20px'})
        ])
    ])
])

# Callback for the interactive revenue trend chart (Booking Trend)
@dash_app.callback(
    Output('revenue-trend-chart', 'figure'),
    [Input('booking-type-dropdown', 'value')]
)
def update_chart(selected_type):
    filtered_data = df[df['Booking Type'] == selected_type]
    revenue_trend = filtered_data.groupby('Booking Date')['Total Revenue'].sum().reset_index()
    fig = px.line(
        revenue_trend, 
        x='Booking Date', 
        y='Total Revenue',
        title=f"Revenue Trend for {selected_type}",
        labels={'Booking Date': 'Date', 'Total Revenue': 'Revenue ($)'}
    )
    return fig

# Callback for the Forecast Revenue tab
@dash_app.callback(
    Output('forecast-graph', 'figure'),
    [Input('forecast-button', 'n_clicks')],
    [
        State('forecast-days', 'value'),
        State('training-days', 'value'),
        State('forecast-model-dropdown', 'value')
    ]
)
def update_forecast(n_clicks, forecast_days, training_days, forecast_model):
    # If no clicks or no forecast horizon is provided, return empty figure
    if n_clicks == 0 or forecast_days is None or training_days is None:
        return {}

    # Sort the data by date
    daily_revenue_sorted = daily_revenue.sort_values('Booking Date')
    
    # Filter data to use only the last `training_days` for training
    cutoff_date = daily_revenue_sorted['Booking Date'].max() - pd.Timedelta(days=training_days)
    recent_data = daily_revenue_sorted[daily_revenue_sorted['Booking Date'] >= cutoff_date]
    
    # Prepare X and y for training
    X = recent_data['Booking Date'].map(datetime.datetime.toordinal).values.reshape(-1, 1)
    y = recent_data['Total Revenue'].values
    
    # Select the forecasting model
    if forecast_model == 'linear':
        model = LinearRegression()
        model.fit(X, y)
    elif forecast_model == 'poly2':
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
    else:  # 'poly3'
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
    
    # Create future dates based on the forecast horizon
    last_date = recent_data['Booking Date'].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    
    # Predict future revenue
    if forecast_model == 'linear':
        forecast_values = model.predict(future_ordinals)
    else:
        # Polynomial transformation
        future_poly = poly.transform(future_ordinals)
        forecast_values = model.predict(future_poly)
    
    # Create a DataFrame for forecasted data
    forecast_df = pd.DataFrame({
        'Booking Date': future_dates,
        'Forecast Revenue': forecast_values
    })
    
    # Combine historical and forecast data for plotting
    # We'll rename the "Total Revenue" to "Revenue" in the historical data
    # for consistency when combining with forecast data
    recent_data_renamed = recent_data.rename(columns={'Total Revenue': 'Revenue'})
    
    # Convert the entire historical segment used for training to the same columns
    combined_df = pd.concat([
        recent_data_renamed[['Booking Date', 'Revenue']],
        forecast_df.rename(columns={'Forecast Revenue': 'Revenue'})
    ], ignore_index=True)
    
    # Create the figure
    fig = px.line(
        combined_df,
        x='Booking Date',
        y='Revenue',
        title=(
            f"Revenue Forecast for Next {forecast_days} Days "
            f"(Trained on last {training_days} days, Model={forecast_model})"
        ),
        labels={'Booking Date': 'Date', 'Revenue': 'Revenue ($)'}
    )
    
    # Overlay the forecast portion in red
    fig.add_scatter(
        x=forecast_df['Booking Date'],
        y=forecast_df['Forecast Revenue'],
        mode='markers+lines',
        name='Forecast',
        marker=dict(color='red')
    )
    
    return fig

# ----------------------------
# Run the Combined App
# ----------------------------
if __name__ == '__main__':
    server.run(debug=True, port=8050)
