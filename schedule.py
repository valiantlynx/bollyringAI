import pandas as pd
import json
import glob
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CALL_LIMIT_PER_WORKER = 180
start_time = time.time()
logging.info("Loading workers data and calculating recommendation scores...")

# Load workers data and format correctly
workers_df = pd.read_json('extracted/workers.json').transpose().reset_index()
workers_df.columns = ['worker_id', 'name', 'base_salary']

# Load and flatten reports data for worker performance
reports_records = []
file_path = glob.glob('extracted/previous_reports/*.json')[0]  # Only process the first file
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

logging.info("Loading and flattening feature calls data...")

# Load feature calls data
feature_records = []
file_path = glob.glob('extracted/feature_calls/*.json')[0]  # Only process the first file
with open(file_path) as f:
    data = json.load(f)
for location, calls in data.items():
    for call_id, call_info in calls.items():
        call_info['call_id'] = call_id
        call_info['location'] = location
        feature_records.append(call_info)
feature_calls_df = pd.DataFrame(feature_records)

logging.info("Starting call assignments to workers based on difficulty preferences and performance...")

# Assign calls to workers based on difficulty preference and performance
def assign_calls_to_workers(calls_df, workers_df):
    # Initialize the schedule dictionary
    schedule = {worker_id: [] for worker_id in workers_df['worker_id']}
    total_calls = len(calls_df)
    
    # Separate calls by difficulty for prioritized assignment
    hard_calls = calls_df[calls_df['difficulty'] == 'hard']
    medium_calls = calls_df[calls_df['difficulty'] == 'medium']
    easy_calls = calls_df[calls_df['difficulty'] == 'easy']
    
    def assign_calls(calls, difficulty):
        for i, (_, call) in enumerate(calls.iterrows(), 1):
            # Get workers sorted by recommendation score and preferred difficulty
            eligible_workers = workers_df[workers_df['preferred_difficulty'] == difficulty]
            eligible_workers = eligible_workers.sort_values(by='avg_recommendation_score', ascending=False)
            
            assigned = False
            for _, worker in eligible_workers.iterrows():
                worker_id = worker['worker_id']
                
                # Assign call if worker hasn't hit the limit
                if len(schedule[worker_id]) < CALL_LIMIT_PER_WORKER:
                    schedule[worker_id].append(call['call_id'])
                    assigned = True
                    break

            # Fallback to any worker, regardless of preference, if no eligible worker can take the call
            if not assigned:
                fallback_worker = workers_df.sort_values(by='avg_recommendation_score', ascending=False)
                for _, fallback_worker_row in fallback_worker.iterrows():
                    fallback_worker_id = fallback_worker_row['worker_id']
                    if len(schedule[fallback_worker_id]) < CALL_LIMIT_PER_WORKER:
                        schedule[fallback_worker_id].append(call['call_id'])
                        assigned = True
                        logging.warning(f"Call {call['call_id']} assigned to fallback worker {fallback_worker_id} due to all eligible workers reaching call limits.")
                        break

            # Log assignment progress
            if i % 100 == 0 or i == total_calls:
                elapsed = time.time() - start_time
                remaining = (elapsed / i) * (total_calls - i)
                logging.info(f"Assigned {i}/{len(calls)} {difficulty} calls. Estimated time remaining: {remaining:.2f} seconds.")

    # Assign calls by difficulty level based on preference
    assign_calls(hard_calls, 'hard')
    assign_calls(medium_calls, 'medium')
    assign_calls(easy_calls, 'easy')

    return schedule

# Generate the call schedule
call_schedule = assign_calls_to_workers(feature_calls_df, workers_df)

logging.info("Saving generated schedule to 'generated_schedule.json'...")

# Save the schedule in the required format
output_schedule = {worker_id: calls for worker_id, calls in call_schedule.items()}

with open('generated_schedule.json', 'w') as outfile:
    json.dump(output_schedule, outfile, indent=4)

total_time = time.time() - start_time
logging.info(f"Call schedule generated and saved to 'generated_schedule.json' in {total_time:.2f} seconds.")
