import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json
import glob

# Load workers data and format correctly
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load prices.json as a dictionary and convert to DataFrame
with open('prices.json') as f:
    prices_data = json.load(f)
prices_df = pd.DataFrame(list(prices_data.items()), columns=['technical_problem', 'price'])

# Function to load and flatten data from multiple JSON files in a directory
def load_and_flatten_data(directory, key_field):
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

# Load feature calls, previous calls, previous reports, and schedules
feature_calls_df = load_and_flatten_data('extracted/feature_calls', 'call_id')
previous_calls_df = load_and_flatten_data('extracted/previous_calls', 'call_id')

# Load previous reports
reports_records = []
for file_path in glob.glob('extracted/previous_reports/*.json'):
    reports_records.extend(pd.read_json(file_path).to_dict(orient='records'))
reports_df = pd.DataFrame(reports_records)

# Load previous schedules
schedule_records = []
for file_path in glob.glob('extracted/previous_schedules/*.json'):
    with open(file_path) as f:
        schedules_data = json.load(f)
    for worker_id, calls in schedules_data.items():
        for call_id in calls:
            schedule_records.append({'worker_id': worker_id, 'call_id': call_id})
schedules_df = pd.DataFrame(schedule_records)

# Combine prices with calls based on technical problems
feature_calls_df = feature_calls_df.merge(prices_df, on='technical_problem', how='left')
previous_calls_df = previous_calls_df.merge(prices_df, on='technical_problem', how='left')

# Ensure call_time is available by adding a dummy column if missing
if 'call_time' not in feature_calls_df.columns:
    feature_calls_df['call_time'] = np.nan

# Calculate Expected Commission and Profit Discrepancy
difficulty_commission_map = {'hard': 1.2, 'medium': 1.0, 'easy': 0.8}
feature_calls_df['expected_commission'] = feature_calls_df['difficulty'].map(difficulty_commission_map) * feature_calls_df['price']
feature_calls_df['profit_discrepancy'] = feature_calls_df['commission'] - feature_calls_df['expected_commission']

# Setup Dash and Plotly Visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("BollyringAI Call Schedule Insights"),
    
    # Summary metrics section
    html.Div([
        html.H3("Summary Metrics"),
        html.P(id='total_calls'),
        html.P(id='total_workers'),
        html.P(id='average_profit')
    ], style={'margin-bottom': '20px'}),
    
    # Dropdown for choosing difficulty levels
    html.Label("Select Difficulty Level:"),
    dcc.Dropdown(
        id='difficulty_dropdown',
        options=[{'label': k.capitalize(), 'value': k} for k in difficulty_commission_map.keys()],
        value='medium'
    ),
    
    # Additional Filters
    html.Label("Select Worker ID:"),
    dcc.Dropdown(
        id='worker_dropdown',
        options=[{'label': worker, 'value': worker} for worker in workers_df['worker_id']],
        value=workers_df['worker_id'][0]
    ),
    
    # Pie charts for distribution insights
    dcc.Graph(id='difficulty_distribution'),
    dcc.Graph(id='location_distribution'),
    
    # Graphs
    dcc.Graph(id='profit_vs_call_time'),
    dcc.Graph(id='commission_vs_profitability'),
    dcc.Graph(id='recommendation_vs_salary'),
    dcc.Graph(id='location_vs_problem'),
    dcc.Graph(id='call_volume_time_series'),
    dcc.Graph(id='call_load_vs_performance'),
    dcc.Graph(id='problem_type_vs_call_time'),
    dcc.Graph(id='commission_distribution'),
    dcc.Graph(id='profit_discrepancy_by_problem'),
    dcc.Graph(id='hourly_call_volume')
])

@app.callback(
    [
        Output('total_calls', 'children'),
        Output('total_workers', 'children'),
        Output('average_profit', 'children'),
        Output('difficulty_distribution', 'figure'),
        Output('location_distribution', 'figure'),
        Output('profit_vs_call_time', 'figure'),
        Output('commission_vs_profitability', 'figure'),
        Output('recommendation_vs_salary', 'figure'),
        Output('location_vs_problem', 'figure'),
        Output('call_volume_time_series', 'figure'),
        Output('call_load_vs_performance', 'figure'),
        Output('problem_type_vs_call_time', 'figure'),
        Output('commission_distribution', 'figure'),
        Output('profit_discrepancy_by_problem', 'figure'),
        Output('hourly_call_volume', 'figure')
    ],
    [Input('difficulty_dropdown', 'value'), Input('worker_dropdown', 'value')]
)
def update_graphs(selected_difficulty, selected_worker):
    # Summary Metrics
    total_calls = f"Total Calls: {len(feature_calls_df)}"
    total_workers = f"Total Workers: {len(workers_df)}"
    average_profit = f"Average Profit per Call: ${reports_df['call_profit'].mean():.2f}"
    
    # 1. Difficulty Level Distribution
    fig_difficulty = px.pie(
        feature_calls_df, 
        names='difficulty', 
        title='Call Distribution by Difficulty Level'
    )
    
    # 2. Location Distribution of Calls
    fig_location = px.pie(
        feature_calls_df, 
        names='location', 
        title='Call Distribution by Location'
    )

    # Profit vs Call Time for selected worker
    fig1 = px.scatter(
        reports_df[reports_df['worker_id'] == selected_worker],
        x='call_time',
        y='call_profit',
        color='likely_to_recommend',
        title=f'Call Profit vs Call Time for Worker {selected_worker}',
        labels={'call_time': 'Call Time (minutes)', 'call_profit': 'Call Profit'}
    )
    
    # Commission vs Profitability by Difficulty
    fig2 = px.bar(
        feature_calls_df.groupby('difficulty')['profit_discrepancy'].mean().reset_index(),
        x='difficulty',
        y='profit_discrepancy',
        title='Average Profit Discrepancy by Difficulty',
        labels={'difficulty': 'Difficulty Level', 'profit_discrepancy': 'Average Profit Discrepancy'}
    )
    
    # Salary bins for recommendation vs salary range
    salary_bins = list(range(0, 31000, 1000))
    salary_labels = [f'{i}-{i+1000}k' for i in range(0, 30000, 1000)]
    worker_recommendation = reports_df.merge(workers_df, on='worker_id')
    worker_recommendation['salary_range'] = pd.cut(worker_recommendation['base_salary'], bins=salary_bins, labels=salary_labels)

    fig3 = px.box(
        worker_recommendation,
        x='salary_range',
        y='likely_to_recommend',
        title='Likely to Recommend vs Worker Salary Range',
        labels={'salary_range': 'Worker Salary Range', 'likely_to_recommend': 'Likely to Recommend'}
    )

    # Call Distribution by Location and Problem Type
    fig4 = px.histogram(
        feature_calls_df,
        x='location',
        color='technical_problem',
        title='Call Distribution by Location and Problem Type',
        labels={'location': 'Location', 'technical_problem': 'Technical Problem'},
        barmode='stack'
    )
    
    # Time Series of Call Volume
    if 'date' in feature_calls_df.columns:
        call_volume_df = feature_calls_df.groupby('date').size().reset_index(name='call_volume')
        fig5 = px.line(
            call_volume_df,
            x='date',
            y='call_volume',
            title='Call Volume Over Time',
            labels={'date': 'Date', 'call_volume': 'Call Volume'}
        )
    else:
        fig5 = px.line(title="No date information available for Call Volume")

    # Worker Call Load vs Performance
    worker_performance = reports_df.groupby('worker_id').agg(
        call_count=('call_id', 'count'),
        avg_profit=('call_profit', 'mean')
    ).reset_index()
    fig6 = px.scatter(
        worker_performance,
        x='call_count',
        y='avg_profit',
        title='Worker Call Load vs Average Profit',
        labels={'call_count': 'Number of Calls', 'avg_profit': 'Average Profit per Call'}
    )
    
    # Problem Type vs Average Call Time
    if 'call_time' in feature_calls_df.columns:
        avg_call_time = feature_calls_df.groupby('technical_problem')['call_time'].mean().reset_index()
        fig7 = px.bar(
            avg_call_time,
            x='technical_problem',
            y='call_time',
            title='Average Call Time by Technical Problem Type',
            labels={'technical_problem': 'Technical Problem', 'call_time': 'Average Call Time'}
        )
    else:
        fig7 = px.bar(title="No call time information available for Problem Type")

    # Commission Distribution Across Difficulty Levels
    fig8 = px.box(
        feature_calls_df,
        x='difficulty',
        y='commission',
        title='Commission Distribution by Difficulty Level',
        labels={'difficulty': 'Difficulty Level', 'commission': 'Commission'}
    )

    # Profit Discrepancy by Problem Type
    fig9 = px.bar(
        feature_calls_df.groupby('technical_problem')['profit_discrepancy'].mean().reset_index(),
        x='technical_problem',
        y='profit_discrepancy',
        title='Average Profit Discrepancy by Problem Type',
        labels={'technical_problem': 'Technical Problem', 'profit_discrepancy': 'Average Profit Discrepancy'}
    )

    # Hourly Call Volume
    if 'date' in feature_calls_df.columns:
        feature_calls_df['hour'] = pd.to_datetime(feature_calls_df['date']).dt.hour
        hourly_volume_df = feature_calls_df.groupby('hour').size().reset_index(name='call_volume')
        fig10 = px.bar(
            hourly_volume_df,
            x='hour',
            y='call_volume',
            title='Hourly Call Volume',
            labels={'hour': 'Hour of Day', 'call_volume': 'Call Volume'}
        )
    else:
        fig10 = px.bar(title="No hourly information available for Call Volume")

    return total_calls, total_workers, average_profit, fig_difficulty, fig_location, fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
