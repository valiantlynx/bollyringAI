import pandas as pd
import json
import glob
import logging
import time
import os
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CALL_LIMIT_PER_WORKER = 267

# Load workers data and format correctly
logging.info("Loading workers data and calculating recommendation scores...")
workers_df = pd.read_json('Schedule1/extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load and flatten reports data for worker performance
reports_records = []
for file_path in glob.glob('Schedule1/extracted/previous_reports/*.json'):
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

# Initialize worker locations (can be inferred from call assignments)
worker_locations = {}

# Load the calls data
calls_data = {}  # This should contain the call data with locations like 'bangalore'
for file_path in glob.glob('Schedule1/extracted/previous_calls/*.json'):
    with open(file_path, 'r') as file:
        calls_data.update(json.load(file))  # Merge data from multiple schedule files


# Load the schedules data (worker-to-call assignments)
schedule_data = {}
for file_path in glob.glob('Schedule1/extracted/previous_schedules/*.json'):
    with open(file_path, 'r') as file:
        schedule_data.update(json.load(file))  # Merge data from multiple schedule files

# Infer worker locations based on the calls they are assigned
for worker_id, assigned_calls in schedule_data.items():
    # We will assume that the location of the first assigned call defines the worker's location
    call_location = None
    for call_id in assigned_calls:
        # Loop through calls data to find the location for each call
        for location, calls in calls_data.items():
            if call_id in calls:
                call_location = location
                break
        if call_location:
            break  # Break after finding the first location
    if call_location:
        worker_locations[worker_id] = call_location
    else:
        worker_locations[worker_id] = 'unknown'  # Default if no location is found

# Assign calls to workers based on difficulty preference and performance
def assign_calls_to_workers(calls_df, workers_df, worker_locations):
    # Initialize the schedule dictionary
    schedule = {worker_id: [] for worker_id in workers_df['worker_id']}
    
    # Ensure 'difficulty' is present in calls_df
    if 'difficulty' not in calls_df.columns or 'location' not in calls_df.columns:
        logging.error("Columns 'difficulty' or 'location' not found in calls_df")
        return schedule
    
    # Separate calls by difficulty for prioritized assignment
    calls_by_difficulty = {
        'hard': calls_df[calls_df['difficulty'] == 'hard'],
        'medium': calls_df[calls_df['difficulty'] == 'medium'],
        'easy': calls_df[calls_df['difficulty'] == 'easy']
    }

    def assign_call_to_worker(call_id, call_location, difficulty):
        # Filter and sort eligible workers by recommendation score and matching location
        eligible_workers = workers_df[(workers_df['preferred_difficulty'] == difficulty) &
                                      (workers_df['worker_id'].isin(worker_locations.keys()))].sort_values(
            by='avg_recommendation_score', ascending=False
        )

        # Only assign workers whose location matches the call location
        eligible_workers = eligible_workers[eligible_workers['worker_id'].apply(
            lambda worker_id: worker_locations.get(worker_id) == call_location)]
        
        # Try assigning to a preferred eligible worker
        for _, worker in eligible_workers.iterrows():
            worker_id = worker['worker_id']
            if len(schedule[worker_id]) < CALL_LIMIT_PER_WORKER:
                schedule[worker_id].append(call_id)
                return True

        # If no eligible worker found, assign to a random worker within the location
        fallback_workers = eligible_workers.sample(frac=1)  # Shuffle workers randomly
        for _, worker in fallback_workers.iterrows():
            worker_id = worker['worker_id']
            if len(schedule[worker_id]) < CALL_LIMIT_PER_WORKER:
                schedule[worker_id].append(call_id)
                logging.warning(f"Call {call_id} assigned to fallback worker {worker_id} due to location match constraints.")
                return True

        # Log error if no available workers are found
        logging.error(f"No available workers for call {call_id}. This should never happen.")
        return False

    # Assign calls by difficulty level based on preference and location match
    for difficulty, calls in calls_by_difficulty.items():
        for _, call in calls.iterrows():
            call_id = call['call_id']
            call_location = call['location']  # Extract location from the call data
            if not assign_call_to_worker(call_id, call_location, difficulty):
                logging.warning(f"Call {call_id} could not be assigned under normal constraints.")

    return schedule

# Process each feature call file and save the schedules in the desired folder structure
def save_schedule_files(schedule):
    week_number = 1  # Example, can be dynamically assigned
    zip_filename = f"call_schedules_week_{week_number}.zip"

    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        # Save each worker's schedule as a JSON file
        for worker_id, assigned_calls in schedule.items():
            call_schedule = {worker_id: {call_id: {} for call_id in assigned_calls}}  # Placeholder for call details
            json_filename = f"call_schedule_{worker_id}.json"
            zipf.writestr(json_filename, json.dumps(call_schedule, indent=4))
            logging.info(f"Saved schedule for worker {worker_id} to {json_filename}")

# Example of saving schedules
calls_df = pd.DataFrame()  # Placeholder, this should be loaded with actual call data
schedule = assign_calls_to_workers(calls_df, workers_df, worker_locations)
save_schedule_files(schedule)
