import pandas as pd
import json
import logging
import time
import os
from tqdm import tqdm
from collections import defaultdict
import glob
import random
import math

# Configure logging to output to both console and file with appropriate levels
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler for warnings and above
file_handler = logging.FileHandler("call_assignment.log")
file_handler.setLevel(logging.WARNING)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Stream handler for info and above
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Constants
MAX_CALL_LIMIT_PER_WORKER = 80
MIN_CALL_LIMIT_PER_WORKER = 22  # Ensuring each worker gets at least 22 calls
FREE_DAY_SCHEDULE_FILES = [
    'calls_32.json',
    'calls_34.json',
    'calls_36.json',
    'calls_38.json',
    'calls_40.json'
]
MUMBAI_LOCATION = 'mumbai'
FREE_DAY_PERCENTAGE = 0.75  # 50%
total_calls_processed = 0


def load_workers_data():
    """Load and format workers data."""
    logging.info("Loading workers data...")
    try:
        with open('worker_performance.json', 'r') as f:
            workers_data = json.load(f)

        # Convert the nested JSON to a DataFrame
        records = []
        for worker_id, details in workers_data.items():
            record = {
                'worker_id': worker_id,
                'name': details.get('worker_name'),
                'location': details.get('location')
            }
            records.append(record)

        workers_df = pd.DataFrame(records)

        # Check for required columns
        expected_columns = {'worker_id', 'name', 'location'}
        actual_columns = set(workers_df.columns)
        missing_columns = expected_columns - actual_columns
        if missing_columns:
            logging.error(f"Missing columns in worker_performance.json: {missing_columns}")
            return pd.DataFrame()

        return workers_df.sort_values(by='name')  # Sorting by name for consistency
    except Exception as e:
        logging.error(f"Failed to load workers data: {e}")
        return pd.DataFrame()


def load_previous_calls():
    """Load and process previous calls data."""
    logging.info("Loading previous calls data...")
    calls_data = {}
    for file_path in glob.glob('Week2/calls/*.json'):
        try:
            with open(file_path, 'r') as file:
                file_data = json.load(file)
                for location, calls in file_data.items():
                    if not isinstance(calls, dict):
                        logging.warning(
                            f"Invalid format for calls in location '{location}' in file {file_path}. Skipping these calls.")
                        continue
                    if location in calls_data:
                        calls_data[location].update(calls)
                    else:
                        calls_data[location] = calls
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error loading {file_path}: {e}. Skipping this file.")
            continue

    # Process calls into DataFrame
    calls_list = []
    for location, calls in calls_data.items():
        for call_id, call_details in calls.items():
            if not isinstance(call_details, dict):
                logging.warning(
                    f"Invalid call details for call ID '{call_id}' in location '{location}'. Skipping this call.")
                continue
            # Ensure required fields are present
            required_fields = ['difficulty', 'technical_problem']
            if not all(field in call_details for field in required_fields):
                logging.warning(
                    f"Missing required fields in call ID '{call_id}' in location '{location}'. Skipping this call.")
                continue
            call_entry = {**call_details, 'call_id': call_id, 'location': location}
            calls_list.append(call_entry)
    df = pd.DataFrame(calls_list)
    logging.info(f"Loaded {len(df)} valid calls from previous data.")
    return df


def load_previous_schedules():
    """Load previous schedule data."""
    logging.info("Loading previous schedules...")
    schedule_data = {}
    for file_path in glob.glob('Week2/call_schedule_Uke2/*.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, dict):
                    logging.warning(f"Invalid schedule format in file {file_path}. Skipping.")
                    continue
                schedule_data.update(data)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error loading {file_path}: {e}. Skipping this file.")
            continue
    logging.info(f"Loaded previous schedule data for {len(schedule_data)} workers.")
    return schedule_data


def determine_worker_locations(workers_df, previous_calls_df, previous_schedule_data):
    """Determine worker locations based on previous assignments."""
    logging.info("Determining worker locations...")
    call_location_dict = previous_calls_df.set_index('call_id')['location'].to_dict()
    worker_locations = {}

    for worker_id, assigned_calls in previous_schedule_data.items():
        worker_call_locations = {call_location_dict.get(call_id) for call_id in assigned_calls}
        worker_call_locations.discard(None)  # Remove None if any call_id not found
        if len(worker_call_locations) == 1:
            worker_locations[worker_id] = next(iter(worker_call_locations))
        elif len(worker_call_locations) > 1:
            worker_locations[worker_id] = 'mixed'
        else:
            worker_locations[worker_id] = 'unknown'

    workers_df['location'] = workers_df['worker_id'].map(worker_locations)

    # Define all_locations to include both previous and current locations
    previous_locations = previous_calls_df['location'].unique().tolist()
    current_locations = ['hyderabad']  # Add other Week3 locations here if any
    all_locations = list(set(previous_locations + current_locations))

    workers_df['locations'] = workers_df['location'].apply(
        lambda x: [x] if x not in ['mixed', 'unknown'] else all_locations
    )

    # Log workers with 'unknown' or 'mixed' locations
    mixed_workers = workers_df[workers_df['location'].isin(['mixed', 'unknown'])]
    logging.info(f"Number of 'mixed' or 'unknown' workers: {len(mixed_workers)}")

    # Log number of workers per location
    for loc in all_locations:
        workers_in_loc = workers_df[workers_df['locations'].apply(lambda locs: loc in locs)]['worker_id'].tolist()
        logging.info(f"Location '{loc}': {len(workers_in_loc)} workers")

    return workers_df


def load_reports_data():
    """Load and process reports data."""
    logging.info("Loading reports data...")
    reports_records = []
    for file_path in glob.glob('Week2/call_report_week2/*.json'):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if not isinstance(data, list):
                    logging.warning(f"Invalid report format in file {file_path}. Skipping.")
                    continue
                reports_records.extend(data)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error loading {file_path}: {e}. Skipping this file.")
            continue
    if reports_records:
        df = pd.DataFrame(reports_records)
        if 'professional_score' in df.columns:
            df = df.drop(['professional_score'], axis=1)
        logging.info(f"Loaded {len(df)} valid report records.")
        return df
    else:
        logging.warning("No valid report records found.")
        return pd.DataFrame()


def calculate_worker_metrics(reports_df, workers_df, previous_calls_df):
    """Calculate comprehensive worker performance metrics."""
    logging.info("Calculating worker performance metrics...")

    if reports_df.empty or workers_df.empty or previous_calls_df.empty:
        logging.error("One or more dataframes are empty. Cannot calculate worker metrics.")
        return pd.DataFrame()

    # Merge dataframes
    merged_df = pd.merge(reports_df, workers_df, on='worker_id', how='left')
    merged_df = pd.merge(
        merged_df,
        previous_calls_df[['call_id', 'technical_problem']],
        on='call_id',
        how='left'
    )

    # Calculate average call times by problem
    avg_call_time = merged_df.groupby(['worker_id', 'technical_problem'])['call_time'].mean()
    avg_call_time_pivot = avg_call_time.unstack(fill_value=0)
    avg_call_time_pivot.columns = [f'avg_{col}_call_time' for col in avg_call_time_pivot.columns]

    # Calculate overall metrics
    overall_metrics = merged_df.groupby('worker_id').agg({
        'call_time': 'mean',
        'likely_to_recommend': 'mean',
        'call_profit': 'sum'
    })
    overall_metrics.columns = ['avg_overall_call_time', 'avg_recommendation_score', 'total_profit']

    # Calculate PPT (Profit Per Time) by technical problem
    profit_time_metrics = merged_df.groupby(['worker_id', 'technical_problem']).agg({
        'call_profit': 'sum',
        'call_time': 'sum'
    })
    # Handle division by zero
    profit_time_metrics['PPT'] = profit_time_metrics.apply(
        lambda row: row['call_profit'] / row['call_time'] if row['call_time'] > 0 else 0,
        axis=1
    )
    ppt_pivot = profit_time_metrics['PPT'].unstack(fill_value=0)
    ppt_pivot.columns = [f'PPT_{col}' for col in ppt_pivot.columns]

    # Combine all metrics
    final_metrics = pd.concat([
        avg_call_time_pivot,
        overall_metrics,
        ppt_pivot
    ], axis=1).reset_index()

    return pd.merge(workers_df, final_metrics, on='worker_id', how='left')


def precompute_worker_rankings(worker_metrics, technical_problems):
    """Precompute worker rankings for all technical problems."""
    logging.info("Precomputing worker rankings...")
    rankings = {}

    for problem in technical_problems:
        ppt_col = f'PPT_{problem}'
        rankings[problem] = defaultdict(list)

        # Group workers by location and sort by PPT
        for location in worker_metrics['location'].unique():
            location_workers = worker_metrics[worker_metrics['location'] == location]
            if ppt_col in location_workers.columns:
                sorted_workers = location_workers.sort_values(by=ppt_col, ascending=False)
                rankings[problem][location] = sorted_workers['worker_id'].tolist()
            else:
                # If no PPT data for the problem, sort workers by total_profit as a fallback
                sorted_workers = location_workers.sort_values(by='total_profit', ascending=False)
                rankings[problem][location] = sorted_workers['worker_id'].tolist()

    return rankings


def calculate_priority_order(worker_metrics, technical_problems):
    """Calculate priority order for technical problems."""
    logging.info("Calculating problem priority order...")
    problem_priorities = []

    for problem in technical_problems:
        ppt_col = f'PPT_{problem}'
        if ppt_col in worker_metrics.columns:
            avg_ppt = worker_metrics[ppt_col].nlargest(30).mean()
        else:
            avg_ppt = 0
        problem_priorities.append({
            'technical_problem': problem,
            'avg_ppt_top_30': avg_ppt
        })

    return pd.DataFrame(problem_priorities).sort_values(
        by='avg_ppt_top_30',
        ascending=False
    ).reset_index(drop=True)


def assign_calls_efficiently(calls_df, workers_df, priority_order_df, worker_rankings, excluded_workers=[]):
    """
    Assign calls efficiently while prioritizing hard and high-value calls first.
    Ensures each worker gets at least MIN_CALL_LIMIT_PER_WORKER calls.
    """
    schedule = defaultdict(list)
    worker_call_counts = defaultdict(int)

    # Define difficulty scores
    difficulties = {'hard': 3, 'medium': 2, 'easy': 1}
    
    # Get problem priority scores from priority_order_df
    known_problems = set(priority_order_df['technical_problem'])
    max_priority = len(priority_order_df)
    problem_priority = {}
    
    # Assign priorities to known problems
    for idx, row in priority_order_df.iterrows():
        problem_priority[row['technical_problem']] = max_priority - idx
    
    # Assign lowest priority to unknown problems
    for problem in calls_df['technical_problem'].unique():
        if problem not in known_problems:
            problem_priority[problem] = 0
            logging.warning(f"Unknown problem type encountered: {problem}")

    # Take only top 120,000 calls proportionally
    total_calls_limit = int(float((120000 * 100 / 260000)/100) * len(calls_df))
    
    # Calculate call scores and create prioritized list
    call_scores = []
    for _, row in calls_df.iterrows():
        call_score = (
            difficulties.get(row['difficulty'], 1) * 1000 +  # Difficulty is primary factor
            problem_priority.get(row['technical_problem'], 0) * 100  # Problem type is secondary factor
        )
        call_scores.append((call_score, row.to_dict()))
    
    # Sort calls by score in descending order
    call_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take only top calls based on the calculated limit
    prioritized_calls = call_scores[:total_calls_limit]
    
    # Convert back to DataFrame for processing
    prioritized_df = pd.DataFrame([call[1] for call in prioritized_calls])
    
    # Organize calls by location
    calls_by_location = defaultdict(list)
    for _, call in prioritized_df.iterrows():
        calls_by_location[call['location']].append(call.to_dict())
    
    # Phase 1: Assign minimum required calls to each worker within their location
    logging.info("Phase 1: Assigning minimum required calls to each worker...")
    for location, calls in calls_by_location.items():
        workers_in_loc = [w for w in workers_df['worker_id']
                         if (location in workers_df.loc[workers_df['worker_id'] == w, 'locations'].values[0])
                         and (w not in excluded_workers)]
        
        if not workers_in_loc:
            logging.warning(f"No workers available for location '{location}'. Skipping all calls in this location.")
            continue
            
        # Calculate minimum calls requirement
        required_calls = len(workers_in_loc) * MIN_CALL_LIMIT_PER_WORKER
        available_calls = len(calls)
        min_calls = min(MIN_CALL_LIMIT_PER_WORKER, max(1, available_calls // len(workers_in_loc)))
        
        # Assign minimum calls using ranked workers
        for _ in range(min_calls):
            for worker in workers_in_loc:
                if not calls:
                    break
                if worker_call_counts[worker] < MAX_CALL_LIMIT_PER_WORKER:
                    call = calls.pop(0)
                    schedule[worker].append(call['call_id'])
                    worker_call_counts[worker] += 1
    
    # Phase 2: Assign remaining calls based on priority and worker ranking
    logging.info("Phase 2: Assigning remaining prioritized calls...")
    for location, calls in calls_by_location.items():
        workers_in_loc = [w for w in workers_df['worker_id']
                         if (location in workers_df.loc[workers_df['worker_id'] == w, 'locations'].values[0])
                         and (w not in excluded_workers)]
        
        for call in calls:
            problem = call['technical_problem']
            call_id = call['call_id']
            
            # Get ranked workers for this problem and location
            ranked_workers = []
            if problem in worker_rankings:
                ranked_workers = [w for w in worker_rankings[problem].get(location, [])
                                if w in workers_in_loc and worker_call_counts[w] < MAX_CALL_LIMIT_PER_WORKER]
            
            if not ranked_workers:
                # Use any available worker in location if no ranked workers found
                ranked_workers = [w for w in workers_in_loc if worker_call_counts[w] < MAX_CALL_LIMIT_PER_WORKER]
            
            if not ranked_workers:
                continue
                
            # Find worker with least calls among ranked workers
            min_calls = min(worker_call_counts[w] for w in ranked_workers)
            eligible_workers = [w for w in ranked_workers if worker_call_counts[w] == min_calls]
            
            # Randomly select one worker among eligible workers
            best_worker = random.choice(eligible_workers)
            schedule[best_worker].append(call_id)
            worker_call_counts[best_worker] += 1

    # Log assignment statistics
    logging.info("Assignment Statistics:")
    for location in calls_by_location.keys():
        workers_in_loc = [w for w in workers_df['worker_id']
                         if (location in workers_df.loc[workers_df['worker_id'] == w, 'locations'].values[0])
                         and (w not in excluded_workers)]
        assigned_calls = sum(len(calls) for wid, calls in schedule.items() if wid in workers_in_loc)
        logging.info(f"Location '{location}': Total calls assigned: {assigned_calls}")
        
        if workers_in_loc:
            calls_per_worker = [len(calls) for w, calls in schedule.items() if w in workers_in_loc]
            if calls_per_worker:
                logging.info(f"Location '{location}': Calls per worker (min/max): {min(calls_per_worker)}/{max(calls_per_worker)}")

    # Log difficulty distribution
    logging.info("Calls by difficulty:")
    difficulty_counts = defaultdict(int)
    for _, call in prioritized_df.iterrows():
        difficulty = call.get('difficulty', 'unknown')
        difficulty_counts[difficulty] += 1
    for diff, count in difficulty_counts.items():
        logging.info(f"{diff}: {count}")

    return schedule
def process_call_files(calls_directory, worker_metrics, technical_problems, excluded_workers, excluded_workers_info):
    global total_calls_processed
    """Process all call files in the directory."""
    logging.info("Processing call files...")

    # Precompute rankings and priority order
    worker_rankings = precompute_worker_rankings(worker_metrics, technical_problems)
    priority_order_df = calculate_priority_order(worker_metrics, technical_problems)

    # Ensure the output directory exists
    output_directory = os.path.join('Week3', 'call_schedules')
    os.makedirs(output_directory, exist_ok=True)

    for file_name in tqdm(os.listdir(calls_directory)):
        if not file_name.endswith('.json'):
            continue

        file_path = os.path.join(calls_directory, file_name)
        start_time = time.time()

        # Determine if this schedule file should have excluded workers
        if file_name in FREE_DAY_SCHEDULE_FILES:
            current_excluded_workers = excluded_workers
            logging.info(f"Excluding {len(current_excluded_workers)} workers from {file_name}.")
        else:
            current_excluded_workers = []

        # Load calls data with error handling
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {file_path}: {e}. Skipping this file.")
            continue
        except Exception as e:
            logging.error(f"Unexpected error loading {file_path}: {e}. Skipping this file.")
            continue

        # Create calls DataFrame with error handling for individual calls
        feature_records = []
        for location, calls in data.items():
            if not isinstance(calls, dict):
                logging.warning(
                    f"Invalid format for calls in location '{location}' in file {file_path}. Skipping these calls.")
                continue
            for call_id, call_info in calls.items():
                if not isinstance(call_info, dict):
                    logging.warning(
                        f"Invalid call details for call ID '{call_id}' in location '{location}'. Skipping this call.")
                    continue
                # Ensure required fields are present
                required_fields = ['difficulty', 'technical_problem']
                if not all(field in call_info for field in required_fields):
                    logging.warning(
                        f"Missing required fields in call ID '{call_id}' in location '{location}'. Skipping this call.")
                    continue
                call_entry = {**call_info, 'call_id': call_id, 'location': location}
                feature_records.append(call_entry)

        if not feature_records:
            logging.warning(f"No valid calls found in file {file_path}. Skipping schedule generation for this file.")
            continue

        feature_calls_df = pd.DataFrame(feature_records)
        total_calls_processed += len(feature_calls_df)
        logging.info(f"Loaded {len(feature_calls_df)} valid calls from {file_name}.")

        # Generate and save schedule
        try:
            schedule = assign_calls_efficiently(
                feature_calls_df,
                worker_metrics,
                priority_order_df,
                worker_rankings,
                excluded_workers=current_excluded_workers
            )
        except Exception as e:
            logging.error(f"Error during call assignment for {file_name}: {e}")
            continue

        # Save the schedule to a JSON file
        output_file = f'call_schedule_{file_name}'
        output_path = os.path.join(output_directory, output_file)
        try:
            with open(output_path, 'w') as f:
                json.dump(schedule, f, indent=4)
            logging.info(f"Saved schedule to {output_path}.")
        except Exception as e:
            logging.error(f"Error writing schedule to {output_path}: {e}")
            continue

        logging.info(f"Processed {file_name} in {time.time() - start_time:.2f} seconds")


def main():
    # List of technical problems
    technical_problems = [
        "account_and_security_issues",
        "basic_hardware_troubleshooting",
        "cloud_and_storage_solutions",
        "device_and_peripheral_setup",
        "email_related_issues",
        "internet_problems",
        "operating_system_support",
        "software_installation_and_configuration",
        "teams_problems",
        "zoom_problems",
        "browser_and_web_based_support",
    ]

    # Load and process all data
    workers_df = load_workers_data()
    if workers_df.empty:
        logging.error("Workers dataframe is empty. Exiting the script.")
        return

    previous_calls_df = load_previous_calls()
    previous_schedule_data = load_previous_schedules()

    # Determine worker locations
    workers_df = determine_worker_locations(workers_df, previous_calls_df, previous_schedule_data)

    # Load and process reports
    reports_df = load_reports_data()

    # Calculate worker metrics
    worker_metrics = calculate_worker_metrics(reports_df, workers_df, previous_calls_df)

    # Check if worker_metrics is valid
    if worker_metrics.empty:
        logging.error("Worker metrics dataframe is empty. Exiting the script.")
        return

    # Select excluded workers dynamically based on call volume
    logging.info(f"Selecting {FREE_DAY_PERCENTAGE * 100}% of Mumbai workers for exclusion from specific schedules...")
    mumbai_workers = worker_metrics[worker_metrics['location'] == MUMBAI_LOCATION]
    total_mumbai_workers = len(mumbai_workers)

    # Calculate the number of workers to exclude (50%)
    excluded_workers_count = math.ceil(total_mumbai_workers * FREE_DAY_PERCENTAGE)

    logging.info(f"Total Mumbai workers: {total_mumbai_workers}")
    logging.info(f"Workers to be excluded (50%): {excluded_workers_count}")

    # Select excluded workers
    if excluded_workers_count > 0:
        selected_free_workers = mumbai_workers.sample(n=excluded_workers_count, random_state=42)
        excluded_worker_ids = selected_free_workers['worker_id'].tolist()
        excluded_workers_info = selected_free_workers[['worker_id', 'name']]
    else:
        excluded_worker_ids = []
        excluded_workers_info = pd.DataFrame(columns=['worker_id', 'name'])
        logging.warning("No workers need to be excluded based on call volume.")

    logging.info(f"Selected {len(excluded_worker_ids)} Mumbai workers for exclusion.")

    # Save excluded workers info to CSV
    if not excluded_workers_info.empty:
        excluded_workers_info.to_csv('excluded_workers_report.csv', index=False)
        logging.info("Saved excluded workers information to 'excluded_workers_report.csv'.")
    else:
        logging.info("No excluded workers to save.")

    # Process call files with excluded workers on specific schedules
    calls_directory = 'Week3/calls/'
    process_call_files(
        calls_directory,
        worker_metrics,
        technical_problems,
        excluded_workers=excluded_worker_ids,
        excluded_workers_info=excluded_workers_info
    )

    # Final summary
    logging.info(f"Total calls processed across all files: {total_calls_processed}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total calls processed: {total_calls_processed}")
    logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")
