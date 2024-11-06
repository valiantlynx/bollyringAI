import json
import pandas as pd
import glob
import numpy as np
import time

# Step 1: Data Loading and Preprocessing
print("Starting data loading and preprocessing...")
start_time = time.time()

# Load workers data
workers_df = pd.read_json('workers.json', orient='index')
workers_df.reset_index(inplace=True)
workers_df.rename(columns={'index': 'worker_id'}, inplace=True)

# Load and combine call schedules (historical)
call_schedule_files = glob.glob('call_shedule_*.json')
schedule_data = []
for file in call_schedule_files:
    with open(file, 'r') as f:
        schedule = json.load(f)
        for worker_id, call_ids in schedule.items():
            for call_id in call_ids:
                schedule_data.append({'worker_id': worker_id, 'call_id': call_id})
schedule_df = pd.DataFrame(schedule_data)

# Load and combine call reports (historical)
call_report_files = glob.glob('call_report_*.json')
call_report_data = []
for file in call_report_files:
    with open(file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            call_report_data.extend(data)
call_report_df = pd.DataFrame(call_report_data)

# Load historical calls data (calls_1.json to calls_10.json)
historical_calls_files = [f'calls_{i}.json' for i in range(1, 11)]
calls_list = []
for file in historical_calls_files:
    with open(file, 'r') as f:
        calls_data = json.load(f)
        for location, calls in calls_data.items():
            for call_id, call_details in calls.items():
                call_details['call_id'] = call_id
                call_details['location'] = location
                calls_list.append(call_details)
calls_df = pd.DataFrame(calls_list)

print(f"Data loading completed in {time.time() - start_time:.2f} seconds.\n")

# Step 2: Analyzing Historical Data
print("Starting data analysis...")
start_time = time.time()

# Merge historical data
historical_df = schedule_df.merge(workers_df, on='worker_id', how='left')
historical_df = historical_df.merge(call_report_df, on=['worker_id', 'call_id'], how='left')
historical_df = historical_df.merge(calls_df, on='call_id', how='left')

# Clean and preprocess data
# Drop any entries with missing location
historical_df.dropna(subset=['location'], inplace=True)

# Determine worker locations based on historical data
# For each worker, get the most frequent location
worker_locations = historical_df.groupby('worker_id')['location'].agg(lambda x: x.mode()[0]).reset_index()
workers_df = workers_df.merge(worker_locations, on='worker_id', how='left')

# Workers who have no historical location data will be included with a default location if possible
# If no location is available, we will assign them to a default 'unknown' location
workers_df['location'].fillna('unknown', inplace=True)

# Create a mapping from location to list of workers
location_workers = workers_df.groupby('location')['worker_id'].apply(list).to_dict()

print(f"Data analysis completed in {time.time() - start_time:.2f} seconds.\n")

# Step 3: Assigning Calls to Workers for Each Future Calls File
print("Starting call assignments per future calls file...")
start_time = time.time()

for N in range(21, 31):
    future_calls_file = f'calls_{N}.json'
    call_schedule_file = f'call_schedule_{N}.json'

    print(f"Processing {future_calls_file}...")

    # Load future calls data
    with open(future_calls_file, 'r') as f:
        calls_data = json.load(f)

    future_calls_list = []
    for location, calls in calls_data.items():
        for call_id, call_details in calls.items():
            call_details['call_id'] = call_id
            call_details['location'] = location
            future_calls_list.append(call_details)
    future_calls_df = pd.DataFrame(future_calls_list)

    # Assign calls
    call_assignments = {}
    call_counts = {}  # For each worker, count of assigned calls

    # For each location, get the workers in that location
    # Initialize call counts per worker
    location_worker_call_counts = {}
    for location in future_calls_df['location'].unique():
        workers_in_location = location_workers.get(location, [])
        if not workers_in_location:
            print(f"No workers available in location {location}.")
            continue
        location_worker_call_counts[location] = {worker_id: 0 for worker_id in workers_in_location}

    # For each call, assign to a worker in the same location
    for idx, call in future_calls_df.iterrows():
        call_id = call['call_id']
        call_location = call['location']
        workers_in_location = location_workers.get(call_location, [])
        if not workers_in_location:
            print(f"No workers available in location {call_location} for call {call_id}.")
            continue
        # Get the worker with the least number of assigned calls in this location
        worker_call_counts = location_worker_call_counts[call_location]
        worker_id = min(worker_call_counts, key=worker_call_counts.get)
        # Assign the call to the worker
        if worker_id not in call_assignments:
            call_assignments[worker_id] = []
        call_assignments[worker_id].append(call_id)
        # Update the call count for the worker
        worker_call_counts[worker_id] += 1

    # Save the call schedule
    with open(call_schedule_file, 'w') as f:
        json.dump(call_assignments, f, indent=4)
    print(f"Assigned calls from {future_calls_file} to {call_schedule_file}.")

print(f"Call assignments completed in {time.time() - start_time:.2f} seconds.\n")
