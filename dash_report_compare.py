import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import json
import glob

# Load workers data
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load prices.json as a dictionary and convert to DataFrame
with open('prices.json') as f:
    prices_data = json.load(f)
prices_df = pd.DataFrame(list(prices_data.items()), columns=['technical_problem', 'price'])

# Function to load and flatten data from multiple JSON files in a directory
def load_and_flatten_data(directory):
    records = []
    for file_path in glob.glob(f'{directory}/*.json'):
        with open(file_path) as f:
            data = json.load(f)
        for location, calls in data.items():
            for call_id, call_info in calls.items():
                call_info['call_id'] = call_id
                call_info['location'] = location
                records.append(call_info)
    return pd.DataFrame(records)

# Load previous and new reports
previous_reports = []
for file_path in glob.glob('extracted/previous_reports/*.json'):
    previous_reports.extend(pd.read_json(file_path).to_dict(orient='records'))
previous_reports_df = pd.DataFrame(previous_reports)

new_reports = []
for file_path in glob.glob('ikkeheltimal_call_reports_11_20/future_call_reports/*.json'):
    new_reports.extend(pd.read_json(file_path).to_dict(orient='records'))
new_reports_df = pd.DataFrame(new_reports)

# Merge reports with location data for city analysis
previous_reports_df = previous_reports_df.merge(load_and_flatten_data('extracted/previous_calls'), on='call_id', how='left')
new_reports_df = new_reports_df.merge(load_and_flatten_data('extracted/feature_calls'), on='call_id', how='left')

# Calculate Sum Profits by city
city_profit_comparison = pd.DataFrame({
    "City": previous_reports_df["location"].unique(),
    "Previous Sum Profit": previous_reports_df.groupby("location")["call_profit"].sum().values,
    "New Sum Profit": new_reports_df.groupby("location")["call_profit"].sum().values
})

# Calculate key performance metrics
def calculate_metrics(report_df):
    return {
        "Sum Profit": report_df['call_profit'].sum(),
        "Average Call Time": report_df['call_time'].mean(),
        "Average Recommendation": report_df['likely_to_recommend'].mean()
    }

# Calculate metrics for previous and new reports
previous_metrics = calculate_metrics(previous_reports_df)
new_metrics = calculate_metrics(new_reports_df)

# Create a DataFrame for overall comparison
comparison_df = pd.DataFrame({
    "Metric": ["Sum Profit", "Average Call Time", "Average Recommendation"],
    "Previous": [previous_metrics["Sum Profit"], previous_metrics["Average Call Time"], previous_metrics["Average Recommendation"]],
    "New": [new_metrics["Sum Profit"], new_metrics["Average Call Time"], new_metrics["Average Recommendation"]],
    "Change (%)": [
        (new_metrics["Sum Profit"] - previous_metrics["Sum Profit"]) / previous_metrics["Sum Profit"] * 100 if previous_metrics["Sum Profit"] else None,
        (new_metrics["Average Call Time"] - previous_metrics["Average Call Time"]) / previous_metrics["Average Call Time"] * 100 if previous_metrics["Average Call Time"] else None,
        (new_metrics["Average Recommendation"] - previous_metrics["Average Recommendation"]) / previous_metrics["Average Recommendation"] * 100 if previous_metrics["Average Recommendation"] else None
    ]
})

# Setup Dash and Plotly Visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Comparison of Previous and New Reports"),
    
    # Comparison Table
    html.Div([
        html.H3("Comparison of Key Metrics"),
        dash_table.DataTable(
            id='comparison_table',
            columns=[{"name": i, "id": i} for i in comparison_df.columns],
            data=comparison_df.to_dict('records'),
            style_table={'width': '40%'}
        )
    ], style={'margin-bottom': '20px'}),
    
    # Comparison Plots
    html.Div([
        dcc.Graph(id='profit_comparison'),
        dcc.Graph(id='call_time_comparison'),
        dcc.Graph(id='recommendation_comparison')
    ]),
    
    # City-level profit comparison plot
    html.Div([
        html.H3("Sum Profit by City"),
        dcc.Graph(id='city_profit_comparison')
    ])
])

# Static callback for initializing figures
@app.callback(
    [
        Output('profit_comparison', 'figure'),
        Output('call_time_comparison', 'figure'),
        Output('recommendation_comparison', 'figure'),
        Output('city_profit_comparison', 'figure')
    ],
    [Input('comparison_table', 'data')]
)
def update_comparison_graphs(data):
    # Profit Comparison
    fig_profit = px.bar(
        comparison_df[comparison_df["Metric"] == "Sum Profit"],
        x="Metric",
        y=["Previous", "New"],
        barmode='group',
        title="Sum Profit Comparison"
    )
    
    # Call Time Comparison
    fig_call_time = px.bar(
        comparison_df[comparison_df["Metric"] == "Average Call Time"],
        x="Metric",
        y=["Previous", "New"],
        barmode='group',
        title="Average Call Time Comparison"
    )
    
    # Recommendation Comparison
    fig_recommendation = px.bar(
        comparison_df[comparison_df["Metric"] == "Average Recommendation"],
        x="Metric",
        y=["Previous", "New"],
        barmode='group',
        title="Average Recommendation Comparison"
    )

    # City Profit Comparison
    fig_city_profit = px.bar(
        city_profit_comparison,
        x="City",
        y=["Previous Sum Profit", "New Sum Profit"],
        barmode='group',
        title="Sum Profit by City"
    )

    return fig_profit, fig_call_time, fig_recommendation, fig_city_profit

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
