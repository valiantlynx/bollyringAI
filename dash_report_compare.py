import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash import dash_table
import json
import glob
import numpy as np

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

# Randomly limit the previous reports to match the size of new reports
limited_previous_reports_df = previous_reports_df.sample(n=len(new_reports_df), random_state=42)

# Calculate metrics
def calculate_metrics(report_df):
    return {
        "Sum Profit": report_df['call_profit'].sum(),
        "Average Call Time": report_df['call_time'].mean(),
        "Average Recommendation": report_df['likely_to_recommend'].mean()
    }

# Calculate metrics for previous, new, and limited previous reports
previous_metrics = calculate_metrics(previous_reports_df)
new_metrics = calculate_metrics(new_reports_df)
limited_previous_metrics = calculate_metrics(limited_previous_reports_df)

# Comparison DataFrames
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

limited_comparison_df = pd.DataFrame({
    "Metric": ["Sum Profit", "Average Call Time", "Average Recommendation"],
    "Limited Previous": [limited_previous_metrics["Sum Profit"], limited_previous_metrics["Average Call Time"], limited_previous_metrics["Average Recommendation"]],
    "New": [new_metrics["Sum Profit"], new_metrics["Average Call Time"], new_metrics["Average Recommendation"]],
    "Change (%)": [
        (new_metrics["Sum Profit"] - limited_previous_metrics["Sum Profit"]) / limited_previous_metrics["Sum Profit"] * 100 if limited_previous_metrics["Sum Profit"] else None,
        (new_metrics["Average Call Time"] - limited_previous_metrics["Average Call Time"]) / limited_previous_metrics["Average Call Time"] * 100 if limited_previous_metrics["Average Call Time"] else None,
        (new_metrics["Average Recommendation"] - limited_previous_metrics["Average Recommendation"]) / limited_previous_metrics["Average Recommendation"] * 100 if limited_previous_metrics["Average Recommendation"] else None
    ]
})

# Sum Profits and Call Counts by City for both comparisons
city_profit_and_count_comparison = pd.DataFrame({
    "City": previous_reports_df["location"].unique(),
    "Previous Sum Profit": previous_reports_df.groupby("location")["call_profit"].sum().values,
    "New Sum Profit": new_reports_df.groupby("location")["call_profit"].sum().values,
    "Previous Call Count": previous_reports_df.groupby("location").size().values,
    "New Call Count": new_reports_df.groupby("location").size().values
})

limited_city_profit_and_count_comparison = pd.DataFrame({
    "City": limited_previous_reports_df["location"].unique(),
    "Limited Previous Sum Profit": limited_previous_reports_df.groupby("location")["call_profit"].sum().values,
    "New Sum Profit": new_reports_df.groupby("location")["call_profit"].sum().values,
    "Limited Previous Call Count": limited_previous_reports_df.groupby("location").size().values,
    "New Call Count": new_reports_df.groupby("location").size().values
})

# Setup Dash and Plotly Visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Comparison of Previous and New Reports"),
    
    # Original Comparison Table
    html.Div([
        html.H3("Original Comparison of Key Metrics"),
        dash_table.DataTable(
            id='comparison_table',
            columns=[{"name": i, "id": i} for i in comparison_df.columns],
            data=comparison_df.to_dict('records'),
            style_table={'width': '40%'}
        )
    ], style={'margin-bottom': '20px'}),
    
    # Grouped Bar Chart with Subplots for Sum Profits and Call Counts by City
    html.Div([
        html.H3("Grouped Bar Chart Comparison of Sum Profit and Call Count by City"),
        dcc.Graph(id='grouped_bar_city_comparison')
    ]),
    
    # Limited Comparison Table
    html.Div([
        html.H3("Comparison with Limited Previous Reports"),
        dash_table.DataTable(
            id='limited_comparison_table',
            columns=[{"name": i, "id": i} for i in limited_comparison_df.columns],
            data=limited_comparison_df.to_dict('records'),
            style_table={'width': '40%'}
        )
    ], style={'margin-bottom': '20px'}),
    
    # Comparison Plots
    html.Div([
        dcc.Graph(id='profit_comparison'),
        dcc.Graph(id='call_time_comparison'),
        dcc.Graph(id='recommendation_comparison')
    ]),
    
    # City-level profit and call count comparison plots
    html.Div([
        html.H3("Original Sum Profit by City"),
        dcc.Graph(id='city_profit_comparison'),
        html.H3("Original Call Count by City"),
        dcc.Graph(id='city_call_count_comparison'),
        html.H3("Limited Sum Profit by City"),
        dcc.Graph(id='limited_city_profit_comparison'),
        html.H3("Limited Call Count by City"),
        dcc.Graph(id='limited_city_call_count_comparison')
    ])
])

# Static callback for initializing figures
@app.callback(
    [
        Output('profit_comparison', 'figure'),
        Output('call_time_comparison', 'figure'),
        Output('recommendation_comparison', 'figure'),
        Output('city_profit_comparison', 'figure'),
        Output('city_call_count_comparison', 'figure'),
        Output('limited_city_profit_comparison', 'figure'),
        Output('limited_city_call_count_comparison', 'figure'),
        Output('grouped_bar_city_comparison', 'figure'),
    ],
    [Input('comparison_table', 'data')]
)
def update_comparison_graphs(data):
    # Original Profit Comparison
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

    # Original City Profit Comparison
    fig_city_profit = px.bar(
        city_profit_and_count_comparison,
        x="City",
        y=["Previous Sum Profit", "New Sum Profit"],
        barmode='group',
        title="Original Sum Profit by City"
    )

    # Original City Call Count Comparison
    fig_city_call_count = px.bar(
        city_profit_and_count_comparison,
        x="City",
        y=["Previous Call Count", "New Call Count"],
        barmode='group',
        title="Original Call Count by City"
    )

    # Limited City Profit Comparison
    fig_limited_city_profit = px.bar(
        limited_city_profit_and_count_comparison,
        x="City",
        y=["Limited Previous Sum Profit", "New Sum Profit"],
        barmode='group',
        title="Limited Sum Profit by City"
    )

    # Limited City Call Count Comparison
    fig_limited_city_call_count = px.bar(
        limited_city_profit_and_count_comparison,
        x="City",
        y=["Limited Previous Call Count", "New Call Count"],
        barmode='group',
        title="Limited Call Count by City"
    )
    
    # Grouped Bar Chart with Subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Sum Profit by City", "Call Count by City")
    )
    
    # Sum Profit Bar Chart
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["Previous Sum Profit"],
            name="Previous Sum Profit",
            marker_color='blue'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["New Sum Profit"],
            name="New Sum Profit",
            marker_color='orange'
        ),
        row=1, col=1
    )
    
    # Call Count Bar Chart
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["Previous Call Count"],
            name="Previous Call Count",
            marker_color='blue'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["New Call Count"],
            name="New Call Count",
            marker_color='orange'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Grouped Bar Chart of Sum Profit and Call Count by City",
        barmode="group"
    )

    return fig_profit, fig_call_time, fig_recommendation, fig_city_profit, fig_city_call_count, fig_limited_city_profit, fig_limited_city_call_count, fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
