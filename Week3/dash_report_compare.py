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
with open('Week1/prices.json') as f:
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

week_1_reports = []
for file_path in glob.glob('Week1/ikkeheltimal_call_reports_11_20/future_call_reports/*.json'):
    week_1_reports.extend(pd.read_json(file_path).to_dict(orient='records'))
week_1_reports_df = pd.DataFrame(week_1_reports)

week_2_reports = []
for file_path in glob.glob('Week2/call_report_week2/*.json'):
    week_2_reports.extend(pd.read_json(file_path).to_dict(orient='records'))
week_2_reports_df = pd.DataFrame(week_2_reports)

new_reports = []
for file_path in glob.glob('Week3/call_reports/*.json'):
    new_reports.extend(pd.read_json(file_path).to_dict(orient='records'))
new_reports_df = pd.DataFrame(new_reports)
print(new_reports_df)

# Merge reports with location data for city analysis
previous_reports_df = previous_reports_df.merge(load_and_flatten_data('extracted/previous_calls'), on='call_id', how='left')
week_1_reports_df = week_1_reports_df.merge(load_and_flatten_data('extracted/feature_calls'), on='call_id', how='left')
week_2_reports_df = week_2_reports_df.merge(load_and_flatten_data('Week2/calls'), on='call_id', how='left')
new_reports_df = new_reports_df.merge(load_and_flatten_data('Week3/calls'), on='call_id', how='left')

# Randomly limit the previous reports to match the size of new reports
limited_previous_reports_df = previous_reports_df.sample(n=len(new_reports_df), random_state=42)

# Use min to handle cases where week_1_reports_df has fewer rows than new_reports_df
limited_week_1_reports_df = week_1_reports_df.sample(n=min(len(week_1_reports_df), len(new_reports_df)), random_state=42)

limited_week_2_reports_df = week_2_reports_df.sample(n=min(len(week_2_reports_df), len(new_reports_df)), random_state=42)


# Load excluded workers from the CSV file
excluded_workers_df = pd.read_csv('excluded_workers_report.csv')

# Create a set of excluded worker IDs for easy lookup
excluded_worker_ids = set(excluded_workers_df['worker_id'])

def calculate_metrics(report_df, workers_df, adjust_for_excluded=False, excluded_worker_ids=None):
    report_df = report_df.merge(workers_df, on='worker_id', how='left')
    
    # Apply salary adjustment only if adjust_for_excluded is True
    if adjust_for_excluded and excluded_worker_ids:
        workers_df['adjusted_salary'] = workers_df.apply(
            lambda row: row['base_salary'] / 2 if row['worker_id'] in excluded_worker_ids else row['base_salary'],
            axis=1
        )
    else:
        workers_df['adjusted_salary'] = workers_df['base_salary']
    
    # Calculate Sum Profit, Average Call Time, Average Recommendation, and Cost
    sum_profit = report_df['call_profit'].sum()
    average_call_time = report_df['call_time'].mean()
    average_recommendation = report_df['likely_to_recommend'].mean()
    cost = workers_df['adjusted_salary'].sum()
    
    # Calculate Profit as Sum Profit - Cost
    profit = sum_profit - cost

    return {
        "Sum Profit": sum_profit,
        "Average Call Time": average_call_time,
        "Average Recommendation": average_recommendation,
        "Cost": cost,
        "Profit": profit
    }

# Calculate metrics for the previous, week 1, week 2, and new reports
previous_metrics = calculate_metrics(previous_reports_df, workers_df)
week_1_metrics = calculate_metrics(week_1_reports_df, workers_df)
week_2_metrics = calculate_metrics(week_2_reports_df, workers_df)
new_metrics = calculate_metrics(new_reports_df, workers_df, adjust_for_excluded=True, excluded_worker_ids=excluded_worker_ids)

limited_previous_metrics = calculate_metrics(limited_previous_reports_df, workers_df, excluded_worker_ids)
limited_week1_metrics = calculate_metrics(limited_week_1_reports_df, workers_df, excluded_worker_ids)
limited_week2_metrics = calculate_metrics(limited_week_2_reports_df, workers_df, excluded_worker_ids)

city_profit_and_count_previous = previous_reports_df.groupby('location').agg(
    total_profit=('call_profit', 'sum'),
    call_time=('call_time', 'sum'),
    call_count=('call_id', 'count'),
    call_recommendation=('likely_to_recommend', 'mean'),
)
city_profit_and_count_week1 = week_1_reports_df.groupby('location').agg(
    total_profit=('call_profit', 'sum'),
    call_time=('call_time', 'sum'),
    call_count=('call_id', 'count'),
    call_recommendation=('likely_to_recommend', 'mean'),
)
city_profit_and_count_week2 = week_2_reports_df.groupby('location').agg(
    total_profit=('call_profit', 'sum'),
    call_time=('call_time', 'sum'),
    call_count=('call_id', 'count'),
    call_recommendation=('likely_to_recommend', 'mean'),
)
city_profit_and_count_new = new_reports_df.groupby('location').agg(
    total_profit=('call_profit', 'sum'),
    call_count=('call_id', 'count')
)

# Create the comparison DataFrame
comparison_df = pd.DataFrame({
    "Metric": ["Sum Profit", "Average Call Time", "Average Recommendation", "Cost", "Profit"],
    "Previous": [previous_metrics["Sum Profit"], previous_metrics["Average Call Time"], previous_metrics["Average Recommendation"], previous_metrics["Cost"], previous_metrics["Profit"]],
    "Week 1": [week_1_metrics["Sum Profit"], week_1_metrics["Average Call Time"], week_1_metrics["Average Recommendation"], week_1_metrics["Cost"], week_1_metrics["Profit"]],
    "Week 2": [week_2_metrics["Sum Profit"], week_2_metrics["Average Call Time"], week_2_metrics["Average Recommendation"], week_2_metrics["Cost"], week_2_metrics["Profit"]],
    "New": [new_metrics["Sum Profit"], new_metrics["Average Call Time"], new_metrics["Average Recommendation"], new_metrics["Cost"], new_metrics["Profit"]],
    "Change (%)": [
        (new_metrics["Sum Profit"] - previous_metrics["Sum Profit"]) / previous_metrics["Sum Profit"] * 100 if previous_metrics["Sum Profit"] else None,
        (new_metrics["Average Call Time"] - previous_metrics["Average Call Time"]) / previous_metrics["Average Call Time"] * 100 if previous_metrics["Average Call Time"] else None,
        (new_metrics["Average Recommendation"] - previous_metrics["Average Recommendation"]) / previous_metrics["Average Recommendation"] * 100 if previous_metrics["Average Recommendation"] else None,
        (new_metrics["Cost"] - previous_metrics["Cost"]) / previous_metrics["Cost"] * 100 if previous_metrics["Cost"] else None,
        (new_metrics["Profit"] - previous_metrics["Profit"]) / previous_metrics["Profit"] * 100 if previous_metrics["Profit"] else None
    ]
})

limited_comparison_df = pd.DataFrame({
    "Metric": ["Sum Profit", "Average Call Time", "Average Recommendation", "Cost", "Profit"],
    "Limited Previous": [
        limited_previous_metrics["Sum Profit"], 
        limited_previous_metrics["Average Call Time"], 
        limited_previous_metrics["Average Recommendation"], 
        limited_previous_metrics["Cost"], 
        limited_previous_metrics["Sum Profit"] - limited_previous_metrics["Cost"]  # Profit calculation for Limited Previous
    ],
    "Week 1": [
        limited_week1_metrics["Sum Profit"], 
        limited_week1_metrics["Average Call Time"], 
        limited_week1_metrics["Average Recommendation"], 
        limited_week1_metrics["Cost"], 
        limited_week1_metrics["Sum Profit"] - limited_week1_metrics["Cost"]  # Profit calculation for Week 1
    ],
    "Week 2": [
        limited_week2_metrics["Sum Profit"], 
        limited_week2_metrics["Average Call Time"], 
        limited_week2_metrics["Average Recommendation"], 
        limited_week2_metrics["Cost"], 
        limited_week2_metrics["Sum Profit"] - limited_week2_metrics["Cost"]  # Profit calculation for Week 2
    ],
    "New": [
        new_metrics["Sum Profit"], 
        new_metrics["Average Call Time"], 
        new_metrics["Average Recommendation"], 
        new_metrics["Cost"], 
        new_metrics["Sum Profit"] - new_metrics["Cost"]  # Profit calculation for New
    ],
    "Change (%)": [
        (new_metrics["Sum Profit"] - limited_previous_metrics["Sum Profit"]) / limited_previous_metrics["Sum Profit"] * 100 if limited_previous_metrics["Sum Profit"] else None,
        (new_metrics["Average Call Time"] - limited_previous_metrics["Average Call Time"]) / limited_previous_metrics["Average Call Time"] * 100 if limited_previous_metrics["Average Call Time"] else None,
        (new_metrics["Average Recommendation"] - limited_previous_metrics["Average Recommendation"]) / limited_previous_metrics["Average Recommendation"] * 100 if limited_previous_metrics["Average Recommendation"] else None,
        (new_metrics["Cost"] - limited_previous_metrics["Cost"]) / limited_previous_metrics["Cost"] * 100 if limited_previous_metrics["Cost"] else None,
        ((new_metrics["Sum Profit"] - new_metrics["Cost"]) - (limited_previous_metrics["Sum Profit"] - limited_previous_metrics["Cost"])) / (limited_previous_metrics["Sum Profit"] - limited_previous_metrics["Cost"]) * 100 if limited_previous_metrics["Sum Profit"] and limited_previous_metrics["Cost"] else None
    ]
})


# Ensure consistent indexing for previous and new data
previous_city_stats = previous_reports_df.groupby("location").agg(
    Previous_Sum_Profit=("call_profit", "sum"),
    Previous_Call_Count=("call_id", "size")
)

week_1_city_stats = week_1_reports_df.groupby("location").agg(
    Week_1_Sum_Profit=("call_profit", "sum"),
    Week_1_Call_Count=("call_id", "size")
)
week_2_city_stats = week_2_reports_df.groupby("location").agg(
    Week_2_Sum_Profit=("call_profit", "sum"),
    Week_2_Call_Count=("call_id", "size")
)
new_city_stats = new_reports_df.groupby("location").agg(
    New_Sum_Profit=("call_profit", "sum"),
    New_Call_Count=("call_id", "size")
)

# Align previous and new data on location (city) to create the comparison DataFrame
city_profit_and_count_comparison = pd.DataFrame({
    "City": previous_city_stats.index.union(new_city_stats.index),
    "Previous Sum Profit": previous_city_stats["Previous_Sum_Profit"].reindex(previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Week 1 Sum Profit": week_1_city_stats["Week_1_Sum_Profit"].reindex(week_1_city_stats.index.union(week_1_city_stats.index), fill_value=0).values,
    "Week 2 Sum Profit": week_2_city_stats["Week_2_Sum_Profit"].reindex(week_2_city_stats.index.union(week_2_city_stats.index), fill_value=0).values,
    "New Sum Profit": new_city_stats["New_Sum_Profit"].reindex(previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Previous Call Count": previous_city_stats["Previous_Call_Count"].reindex(previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Week 1 Call Count": week_1_city_stats["Week_1_Call_Count"].reindex(week_1_city_stats.index.union(week_1_city_stats.index), fill_value=0).values,
    "Week 2 Call Count": week_2_city_stats["Week_2_Call_Count"].reindex(week_2_city_stats.index.union(week_2_city_stats.index), fill_value=0).values,
    "New Call Count": new_city_stats["New_Call_Count"].reindex(previous_city_stats.index.union(new_city_stats.index), fill_value=0).values
})

# Ensure consistent indexing for limited previous and new data
limited_previous_city_stats = limited_previous_reports_df.groupby("location").agg(
    Limited_Previous_Sum_Profit=("call_profit", "sum"),
    Limited_Previous_Call_Count=("call_id", "size")
)

limited_week_1_city_stats = limited_week_1_reports_df.groupby("location").agg(
    Limited_Week_1_Sum_Profit=("call_profit", "sum"),
    Limited_Week_1_Call_Count=("call_id", "size")
)
limited_week_2_city_stats = limited_week_2_reports_df.groupby("location").agg(
    Limited_Week_2_Sum_Profit=("call_profit", "sum"),
    Limited_Week_2_Call_Count=("call_id", "size")
)

# Align limited previous and new data on location (city) to create the comparison DataFrame
limited_city_profit_and_count_comparison = pd.DataFrame({
    "City": limited_previous_city_stats.index.union(new_city_stats.index),
    "Limited Previous Sum Profit": limited_previous_city_stats["Limited_Previous_Sum_Profit"].reindex(limited_previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Limited Week 1 Sum Profit": limited_week_1_city_stats["Limited_Week_1_Sum_Profit"].reindex(limited_week_1_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Limited Week 2 Sum Profit": limited_week_2_city_stats["Limited_Week_2_Sum_Profit"].reindex(limited_week_2_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "New Sum Profit": new_city_stats["New_Sum_Profit"].reindex(limited_previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Limited Previous Call Count": limited_previous_city_stats["Limited_Previous_Call_Count"].reindex(limited_previous_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Limited Week 1 Call Count": limited_week_1_city_stats["Limited_Week_1_Call_Count"].reindex(limited_week_1_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "Limited Week 2 Call Count": limited_week_2_city_stats["Limited_Week_2_Call_Count"].reindex(limited_week_2_city_stats.index.union(new_city_stats.index), fill_value=0).values,
    "New Call Count": new_city_stats["New_Call_Count"].reindex(limited_previous_city_stats.index.union(new_city_stats.index), fill_value=0).values
})

print# Print city profit and call counts for verification
print("----------------- - Previous:")
print(city_profit_and_count_previous)
print(city_profit_and_count_comparison)
print("\n---------------- - New:")
print(city_profit_and_count_new)
print(limited_city_profit_and_count_comparison)


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
    
    # Limited Grouped Bar Chart with Subplots for Sum Profits and Call Counts by City
    html.Div([
        html.H3("Limited Grouped Bar Chart Comparison of Sum Profit and Call Count by City"),
        dcc.Graph(id='limited_grouped_bar_city_comparison')
    ]),
    
    # Comparison Plots
    html.Div([
        dcc.Graph(id='profit_comparison'),
        dcc.Graph(id='cost_comparison'),
        dcc.Graph(id='single_profit_comparison'),
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
        Output('cost_comparison', 'figure'),
        Output('single_profit_comparison', 'figure'),
        Output('call_time_comparison', 'figure'),
        Output('recommendation_comparison', 'figure'),
        Output('city_profit_comparison', 'figure'),
        Output('city_call_count_comparison', 'figure'),
        Output('limited_city_profit_comparison', 'figure'),
        Output('limited_city_call_count_comparison', 'figure'),
        Output('grouped_bar_city_comparison', 'figure'),
        Output('limited_grouped_bar_city_comparison', 'figure'),
    ],
    [Input('comparison_table', 'data')]
)
def update_comparison_graphs(data):
    # Original Profit Comparison
    fig_profit = px.bar(
        comparison_df[comparison_df["Metric"] == "Sum Profit"],
        x="Metric",
        y=["Previous", "Week 1", "Week 2", "New"],
        barmode='group',
        title="Sum Profit Comparison"
    )
    
    fig_cost = px.bar(
        comparison_df[comparison_df["Metric"] == "Cost"],
        x="Metric",
        y=["Previous", "Week 1", "Week 2", "New"],
        barmode='group',
        title="Cost Comparison"
    )
    fig_single_profit = px.bar(
        comparison_df[comparison_df["Metric"] == "Profit"],
        x="Metric",
        y=["Previous", "Week 1", "Week 2", "New"],
        barmode='group',
        title="Profit Difference Comparison"
    )
    # Call Time Comparison
    fig_call_time = px.bar(
        comparison_df[comparison_df["Metric"] == "Average Call Time"],
        x="Metric",
        y=["Previous", "Week 1", "Week 2", "New"],
        barmode='group',
        title="Average Call Time Comparison"
    )
    
    # Recommendation Comparison
    fig_recommendation = px.bar(
        comparison_df[comparison_df["Metric"] == "Average Recommendation"],
        x="Metric",
        y=["Previous", "Week 1", "Week 2", "New"],
        barmode='group',
        title="Average Recommendation Comparison"
    )

    # Original City Profit Comparison
    fig_city_profit = px.bar(
        city_profit_and_count_comparison,
        x="City",
        y=["Previous Sum Profit", "Week 1 Sum Profit", "Week 2 Sum Profit", "New Sum Profit"],
        barmode='group',
        title="Original Sum Profit by City"
    )

    # Original City Call Count Comparison
    fig_city_call_count = px.bar(
        city_profit_and_count_comparison,
        x="City",
        y=["Previous Call Count","Week 1 Call Count","Week 2 Call Count", "New Call Count"],
        barmode='group',
        title="Original Call Count by City"
    )

    # Limited City Profit Comparison
    fig_limited_city_profit = px.bar(
        limited_city_profit_and_count_comparison,
        x="City",
        y=["Limited Previous Sum Profit", "Limited Week 1 Sum Profit","Limited Week 2 Sum Profit", "New Sum Profit"],
        barmode='group',
        title="Limited Sum Profit by City"
    )

    # Limited City Call Count Comparison
    fig_limited_city_call_count = px.bar(
        limited_city_profit_and_count_comparison,
        x="City",
        y=["Limited Previous Call Count", "Limited Week 1 Call Count","Limited Week 2 Call Count", "New Call Count"],
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
            y=city_profit_and_count_comparison["Week 1 Sum Profit"],
            name="Week 1 Sum Profit",
            marker_color='red'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["Week 2 Sum Profit"],
            name="Week 2 Sum Profit",
            marker_color='green'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["New Sum Profit"],
            name="New Sum Profit",
            marker_color='green'
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
            y=city_profit_and_count_comparison["Week 1 Call Count"],
            name="Week 1 Call Count",
            marker_color='red'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["Week 2 Call Count"],
            name="Week 2 Call Count",
            marker_color='green'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(
            x=city_profit_and_count_comparison["City"],
            y=city_profit_and_count_comparison["New Call Count"],
            name="New Call Count",
            marker_color='green'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Grouped Bar Chart of Sum Profit and Call Count by City",
        barmode="group"
    )
    
    # Limited Grouped Bar Chart with Subplots
    fig_limited = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Limited Sum Profit by City", "Limited Call Count by City")
    )
    
    # Sum Profit Bar Chart
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Previous Sum Profit"],
            name="Limited Previous Sum Profit",
            marker_color='blue'
        ),
        row=1, col=1
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Week 1 Sum Profit"],
            name="Limited Week 1 Sum Profit",
            marker_color='red'
        ),
        row=1, col=1
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Week 2 Sum Profit"],
            name="Limited Week 2 Sum Profit",
            marker_color='green'
        ),
        row=1, col=1
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["New Sum Profit"],
            name="New Sum Profit",
            marker_color='green'
        ),
        row=1, col=1
    )
    
    # Call Count Bar Chart
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Previous Call Count"],
            name="Limited Previous Call Count",
            marker_color='blue'
        ),
        row=1, col=2
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Week 1 Call Count"],
            name="Limited Week 1 Call Count",
            marker_color='red'
        ),
        row=1, col=2
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["Limited Week 2 Call Count"],
            name="Limited Week 2 Call Count",
            marker_color='green'
        ),
        row=1, col=2
    )
    fig_limited.add_trace(
        go.Bar(
            x=limited_city_profit_and_count_comparison["City"],
            y=limited_city_profit_and_count_comparison["New Call Count"],
            name="New Call Count",
            marker_color='green'
        ),
        row=1, col=2
    )
    
    fig_limited.update_layout(
        title="Limited Grouped Bar Chart of Sum Profit and Call Count by City",
        barmode="group"
    )

    return fig_profit, fig_cost, fig_single_profit, fig_call_time, fig_recommendation, fig_city_profit, fig_city_call_count, fig_limited_city_profit, fig_limited_city_call_count, fig, fig_limited

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
