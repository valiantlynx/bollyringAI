import pandas as pd
import json
import glob
import logging
import time
import os
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MAX_CALL_LIMIT_PER_WORKER = 80
MIN_CALL_LIMIT_PER_WORKER = 25  # Set to -1 for unlimited calls

# %% Load workers data and format correctly
logging.info("Loading workers data and calculating recommendation scores...")
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']
workers_df.sort_values(by='base_salary')

# %% Process calls data from all files at once
logging.info("Processing calls data...")

calls_data = {}
for file_path in glob.glob('Week2/calls/*.json'):
    with open(file_path, 'r') as file:
        file_data = json.load(file)
        for location, calls in file_data.items():
            if location in calls_data:
                calls_data[location].update(calls)  # Merge calls for this location
            else:
                calls_data[location] = calls  # Initialize calls for this location

# Flatten calls data
calls_list = []
for location, calls in calls_data.items():
    for call_id, call_details in calls.items():
        call_details['call_id'] = call_id
        call_details['location'] = location  # Add location to each call record
        calls_list.append(call_details)

previous_calls_df = pd.DataFrame(calls_list)

# %% Create call location dictionary for fast lookup
call_location_dict = previous_calls_df.set_index('call_id')['location'].to_dict()

# %% Process previous schedule data into a DataFrame for workers' locations
logging.info("Processing previous schedules...")

previous_schedule_data = {}
for file_path in glob.glob('Week2/call_schedule_Uke2/*.json'):
    with open(file_path, 'r') as file:
        previous_schedule_data.update(json.load(file))  # Merge data from multiple schedule files

# Map each worker to their location based on the calls they were assigned
worker_locations = {}

# Loop through each worker's assigned calls and update their location
for worker_id, assigned_calls in previous_schedule_data.items():
    worker_call_locations = set()  # Use set to ensure unique locations
    for call_id in assigned_calls:
        call_location = call_location_dict.get(call_id)  # Fast lookup for the call location
        if call_location:
            worker_call_locations.add(call_location)
    
    # Assign location or mark as mixed
    worker_locations[worker_id] = "mixed" if len(worker_call_locations) > 1 else worker_call_locations.pop()

# Add locations to the workers dataframe
workers_df['location'] = workers_df['worker_id'].map(worker_locations)

# %% Check if there are workers in multiple offices
mixed_location_workers = workers_df[workers_df['location'] == 'mixed']
logging.info(f"Number of workers in multiple locations: {len(mixed_location_workers)}")

# %% Load and flatten reports data for worker performance
logging.info("Loading and flattening report data...")

reports_records = []
for file_path in glob.glob('Week2/call_report_week2/*.json'):
    with open(file_path, 'r') as file:
        reports_records.extend(pd.read_json(file).to_dict(orient='records'))
reports_df = pd.DataFrame(reports_records)

# Remove unnecessary columns
reports_df = reports_df.drop(['professional_score'], axis=1)

# %% Merge reports with workers and previous calls data
logging.info("Merging dataframes...")

# Merge workers_df with reports_df on 'worker_id' to include worker details in the report data
merged_df = pd.merge(reports_df, workers_df[['worker_id', 'location']], on='worker_id', how='left')
# Merge with previous_calls_df to get call details
merged_df = pd.merge(merged_df, previous_calls_df[['call_id', 'location']], on='call_id', how='left', suffixes=('', '_previous'))
# Drop the redundant location_previous column
merged_df = merged_df.drop(columns=['location_previous'])

# %% Optimize the worker-location count by grouping merged dataframe
logging.info("Counting the number of unique workers per location...")

location_worker_counts = merged_df.groupby('location')['worker_id'].nunique()

# %% Display the results
logging.info(f"Worker counts by location:\n{location_worker_counts}")
