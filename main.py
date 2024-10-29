import pandas as pd
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import json

# Load workers data and format correctly
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load prices.json as a dictionary and convert to DataFrame
with open('prices.json') as f:
    prices_data = json.load(f)
prices_df = pd.DataFrame(list(prices_data.items()), columns=['technical_problem', 'price'])

# Load and flatten feature_calls.json
with open('extracted/feature_calls/calls_11.json') as f:
    feature_calls_data = json.load(f)

# Flatten feature_calls_data
feature_records = []
for location, calls in feature_calls_data.items():
    for call_id, call_info in calls.items():
        call_info['call_id'] = call_id
        call_info['location'] = location
        feature_records.append(call_info)
feature_calls_df = pd.DataFrame(feature_records)

# Load and flatten previous_calls.json
with open('extracted/previous_calls/calls_0.json') as f:
    previous_calls_data = json.load(f)
previous_records = []
for location, calls in previous_calls_data.items():
    for call_id, call_info in calls.items():
        call_info['call_id'] = call_id
        call_info['location'] = location
        previous_records.append(call_info)
previous_calls_df = pd.DataFrame(previous_records)

# Load reports data
reports_df = pd.read_json('extracted/previous_reports/call_report_0.json')

# Load and flatten schedules data
with open('extracted/previous_schedules/call_shedule_0.json') as f:
    schedules_data = json.load(f)
schedule_records = []
for worker_id, calls in schedules_data.items():
    for call_id in calls:
        schedule_records.append({'worker_id': worker_id, 'call_id': call_id})
schedules_df = pd.DataFrame(schedule_records)

# Combine prices with calls based on technical problems
feature_calls_df = feature_calls_df.merge(prices_df, on='technical_problem', how='left')
previous_calls_df = previous_calls_df.merge(prices_df, on='technical_problem', how='left')

# Ensure call_time is available by adding a dummy column if missing
if 'call_time' not in feature_calls_df.columns:
    feature_calls_df['call_time'] = np.nan  # Or set this with actual values if available

# Calculate Expected Commission and Profit Discrepancy
difficulty_commission_map = {'hard': 1.2, 'medium': 1.0, 'easy': 0.8}
feature_calls_df['expected_commission'] = feature_calls_df['difficulty'].map(difficulty_commission_map) * feature_calls_df['price']
feature_calls_df['profit_discrepancy'] = feature_calls_df['commission'] - feature_calls_df['expected_commission']

# Setup Dash and Plotly Visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("BollyringAI Call Schedule Insights"),
    
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
    
    # Graphs
    dcc.Graph(id='profit_vs_call_time'),
    dcc.Graph(id='commission_vs_profitability'),
    dcc.Graph(id='recommendation_vs_salary'),
    dcc.Graph(id='location_vs_problem'),
    dcc.Graph(id='call_volume_time_series'),
    dcc.Graph(id='call_load_vs_performance'),
    dcc.Graph(id='problem_type_vs_call_time')
])

@app.callback(
    [
        Output('profit_vs_call_time', 'figure'),
        Output('commission_vs_profitability', 'figure'),
        Output('recommendation_vs_salary', 'figure'),
        Output('location_vs_problem', 'figure'),
        Output('call_volume_time_series', 'figure'),
        Output('call_load_vs_performance', 'figure'),
        Output('problem_type_vs_call_time', 'figure')
    ],
    [Input('difficulty_dropdown', 'value'), Input('worker_dropdown', 'value')]
)
def update_graphs(selected_difficulty, selected_worker):
    # Filter data based on difficulty level
    filtered_df = feature_calls_df[feature_calls_df['difficulty'] == selected_difficulty]
    
    # 1. Profit vs Call Time for selected worker
    fig1 = px.scatter(
        reports_df[reports_df['worker_id'] == selected_worker],
        x='call_time',
        y='call_profit',
        color='likely_to_recommend',
        title=f'Call Profit vs Call Time for Worker {selected_worker}',
        labels={'call_time': 'Call Time (minutes)', 'call_profit': 'Call Profit'}
    )
    
    # 2. Commission vs Profitability by Difficulty
    fig2 = px.bar(
        feature_calls_df.groupby('difficulty')['profit_discrepancy'].mean().reset_index(),
        x='difficulty',
        y='profit_discrepancy',
        title='Average Profit Discrepancy by Difficulty',
        labels={'difficulty': 'Difficulty Level', 'profit_discrepancy': 'Average Profit Discrepancy'}
    )
    
    # 3. Recommendation Score Distribution by Worker Salary
    worker_recommendation = reports_df.merge(workers_df, left_on='worker_id', right_on='worker_id')
    fig3 = px.box(
        worker_recommendation,
        x='base_salary',
        y='likely_to_recommend',
        title='Likely to Recommend vs Worker Salary',
        labels={'base_salary': 'Worker Salary', 'likely_to_recommend': 'Likely to Recommend'}
    )
    
    # 4. Call Distribution by Location and Problem Type
    fig4 = px.histogram(
        feature_calls_df,
        x='location',
        color='technical_problem',
        title='Call Distribution by Location and Problem Type',
        labels={'location': 'Location', 'technical_problem': 'Technical Problem'},
        barmode='stack'
    )
    
    # 5. Time Series of Call Volume
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

    # 6. Worker Call Load vs Performance
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
    
    # 7. Problem Type vs Average Call Time
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

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
