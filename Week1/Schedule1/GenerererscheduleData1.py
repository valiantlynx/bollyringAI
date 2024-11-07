import json
import pandas as pd

# Load data from files
def load_data():
    call_reports = []
    for i in range(10):
        with open(f'./extracted/previous_reports/call_report_{i}.json', 'r') as f:
            call_reports.extend(json.load(f))
    call_report_df = pd.DataFrame(call_reports)

    calls_data = []
    for i in range(10):
        with open(f'./extracted/previous_calls/calls_{i}.json', 'r') as f:
            calls = json.load(f)
            for city, calls_in_city in calls.items():
                for call_id, call_info in calls_in_city.items():
                    call_info.update({"call_id": call_id, "city": city})
                    calls_data.append(call_info)
    calls_df = pd.DataFrame(calls_data)

    with open('./extracted/workers.json', 'r') as f:
        workers_data = json.load(f)
    workers_df = pd.DataFrame.from_dict(workers_data, orient='index').reset_index().rename(
        columns={'index': 'worker_id'})

    return call_report_df, calls_df, workers_df


# Calculate average profit and time per worker for each technical problem
def calculate_worker_performance(call_report_df, calls_df):
    merged_df = pd.merge(call_report_df, calls_df, on='call_id')
    worker_performance = merged_df.groupby(['worker_id']).agg({
        'call_profit': 'mean',
        'call_time': 'mean'
    }).reset_index()
    worker_performance['performance'] = pd.qcut(worker_performance['call_profit'], q=3, labels=['poor', 'average', 'good'])
    return worker_performance


# Classify calls by difficulty
def classify_calls(calls_df):
    calls_df['difficulty_class'] = calls_df['difficulty'].map({
        'hard': 'hard',
        'medium': 'medium',
        'easy': 'easy'
    }).fillna('medium')
    return calls_df


# Assign workers to calls based on performance and call difficulty
def assign_calls_based_on_performance(worker_performance, classified_calls_df):
    assignments = []

    good_workers = worker_performance[worker_performance['performance'] == 'good']['worker_id']
    average_workers = worker_performance[worker_performance['performance'] == 'average']['worker_id']
    poor_workers = worker_performance[worker_performance['performance'] == 'poor']['worker_id']

    for _, call in classified_calls_df.iterrows():
        if call['difficulty_class'] == 'hard':
            assigned_worker = good_workers.sample().values[0]
        elif call['difficulty_class'] == 'medium':
            assigned_worker = average_workers.sample().values[0]
        else:
            assigned_worker = poor_workers.sample().values[0]

        assignments.append({
            'call_id': call['call_id'],
            'worker_id': assigned_worker,
            'predicted_profit': call.get('commission', 0)
        })

    return pd.DataFrame(assignments)


# Save each schedule to a separate JSON file
def save_schedule(assignments, file_index):
    schedule = {}
    for _, row in assignments.iterrows():
        worker_id = row['worker_id']
        call_id = row['call_id']
        if worker_id not in schedule:
            schedule[worker_id] = []
        schedule[worker_id].append(call_id)

    with open(f'optimized_call_schedule_{file_index}.json', 'w') as f:
        json.dump(schedule, f, indent=2)
    print(f"Schedule saved to 'optimized_call_schedule_{file_index}.json' in the correct format.")


# Process each future call file
def process_future_call_files(worker_performance, workers_df):
    for i in range(11,21):
        # Load each future call file
        with open(f'./extracted/feature_calls/calls_{i}.json', 'r') as f:
            future_calls = json.load(f)

        # Convert future calls to DataFrame
        calls_data = []
        for city, calls_in_city in future_calls.items():
            for call_id, call_info in calls_in_city.items():
                call_info.update({"call_id": call_id, "city": city})
                calls_data.append(call_info)
        future_calls_df = pd.DataFrame(calls_data)

        # Classify calls by difficulty
        classified_calls_df = classify_calls(future_calls_df)

        # Assign workers based on performance and call difficulty
        assignments = assign_calls_based_on_performance(worker_performance, classified_calls_df)

        # Save each schedule to a separate JSON file
        save_schedule(assignments, i)


# Main Execution
call_report_df, calls_df, workers_df = load_data()
worker_performance = calculate_worker_performance(call_report_df, calls_df)
process_future_call_files(worker_performance, workers_df)
