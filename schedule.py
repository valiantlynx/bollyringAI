import pandas as pd
import json
import glob
import logging
import time
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CALL_LIMIT_PER_WORKER = 180

# Load workers data and format correctly
logging.info("Loading workers data and calculating recommendation scores...")
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load and flatten reports data for worker performance
reports_records = []
for file_path in glob.glob('extracted/previous_reports/*.json'):
    reports_records.extend(pd.read_json(file_path).to_dict(orient='records'))
reports_df = pd.DataFrame(reports_records)

# Calculate average recommendation score for each worker
worker_performance = reports_df.groupby('worker_id')['likely_to_recommend'].mean().reset_index()
worker_performance.columns = ['worker_id', 'avg_recommendation_score']

# Merge performance data with workers
workers_df = workers_df.merge(worker_performance, on='worker_id', how='left')

# Define thresholds based on recommendation score
def assign_difficulty_preference(score):
    if score > 2.2:
        return 'hard'
    elif 1.8 <= score <= 2.2:
        return 'medium'
    else:
        return 'easy'

# Assign difficulty preference to each worker
workers_df['preferred_difficulty'] = workers_df['avg_recommendation_score'].apply(assign_difficulty_preference)

# Assign calls to workers based on difficulty preference and performance
def assign_calls_to_workers(calls_df, workers_df):
    # Initialize the schedule dictionary
    schedule = {worker_id: [] for worker_id in workers_df['worker_id']}
    
    # Separate calls by difficulty for prioritized assignment
    calls_by_difficulty = {
        'hard': calls_df[calls_df['difficulty'] == 'hard'],
        'medium': calls_df[calls_df['difficulty'] == 'medium'],
        'easy': calls_df[calls_df['difficulty'] == 'easy']
    }

    def assign_call_to_worker(call_id, difficulty):
        # Filter and sort eligible workers by recommendation score
        eligible_workers = workers_df[workers_df['preferred_difficulty'] == difficulty].sort_values(
            by='avg_recommendation_score', ascending=False
        )
        
        # Try assigning to a preferred eligible worker
        for _, worker in eligible_workers.iterrows():
            worker_id = worker['worker_id']
            if len(schedule[worker_id]) < CALL_LIMIT_PER_WORKER:
                schedule[worker_id].append(call_id)
                return True

        # If preferred eligible workers are full, assign to a random available worker under the limit
        fallback_workers = workers_df.sample(frac=1)  # Shuffle to randomize fallback selection
        for _, worker in fallback_workers.iterrows():
            worker_id = worker['worker_id']
            if len(schedule[worker_id]) < CALL_LIMIT_PER_WORKER:
                schedule[worker_id].append(call_id)
                logging.warning(f"Call {call_id} assigned to fallback worker {worker_id} due to all eligible workers reaching call limits.")
                return True

        # Log error if all workers are at their limit
        logging.error(f"No available workers for call {call_id}. This should never happen.")
        return False

    # Assign calls by difficulty level based on preference
    for difficulty, calls in calls_by_difficulty.items():
        for _, call in calls.iterrows():
            call_id = call['call_id']
            if not assign_call_to_worker(call_id, difficulty):
                logging.warning(f"Call {call_id} could not be assigned under normal constraints.")

    return schedule

# Process each feature call file
for file_path in glob.glob('extracted/feature_calls/*.json'):
    start_time = time.time()
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    logging.info(f"Processing feature calls file: {file_path}")
    
    # Load and flatten feature calls data for the current file
    feature_records = []
    with open(file_path) as f:
        data = json.load(f)
    for location, calls in data.items():
        for call_id, call_info in calls.items():
            call_info['call_id'] = call_id
            call_info['location'] = location
            feature_records.append(call_info)
    feature_calls_df = pd.DataFrame(feature_records)

    # Generate the call schedule for this file
    call_schedule = assign_calls_to_workers(feature_calls_df, workers_df)

    # Save the schedule in the required format
    output_schedule = {worker_id: calls for worker_id, calls in call_schedule.items()}
    output_file = f'generated_schedule_{file_name}.json'
    
    with open(output_file, 'w') as outfile:
        json.dump(output_schedule, outfile, indent=4)
    
    total_time = time.time() - start_time
    logging.info(f"Call schedule for {file_name} generated and saved to '{output_file}' in {total_time:.2f} seconds.")
