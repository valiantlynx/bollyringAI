# scripts/calculate_total_commission.py

import os
import json
import glob
import pandas as pd

def load_schedule(schedule_file):
    """
    Load a call schedule JSON file.
    """
    with open(schedule_file, 'r') as f:
        schedule = json.load(f)
    return schedule

def load_calls(calls_file):
    """
    Load calls from the JSON file into a DataFrame.
    """
    with open(calls_file, 'r') as f:
        data = json.load(f)
    calls_data = []
    for location, calls_dict in data.items():
        for call_id, details in calls_dict.items():
            details['call_id'] = call_id
            calls_data.append(details)
    calls_df = pd.DataFrame(calls_data)
    return calls_df

def calculate_total_commission(schedules_dir, calls_dir):
    """
    Calculate the total commission from all calls in all schedules.
    """
    schedule_files = glob.glob(os.path.join(schedules_dir, 'call_schedule_*.json'))
    total_commission = 0.0
    processed_call_ids = set()

    for schedule_file in schedule_files:
        # Extract the schedule number to find the corresponding calls file
        schedule_filename = os.path.basename(schedule_file)
        schedule_number = ''.join(filter(str.isdigit, schedule_filename))
        calls_file = os.path.join(calls_dir, f'calls_{schedule_number}.json')

        if not os.path.exists(calls_file):
            print(f"Calls file '{calls_file}' not found for schedule '{schedule_filename}'. Skipping.")
            continue

        # Load schedule and calls data
        schedule = load_schedule(schedule_file)
        calls_df = load_calls(calls_file)
        if calls_df.empty:
            print(f"No call data available from '{calls_file}'. Skipping.")
            continue

        # Create a mapping from call_id to commission
        calls_df.set_index('call_id', inplace=True)
        call_commissions = calls_df['commission'].to_dict()

        # Collect all call IDs from the schedule
        all_call_ids = []
        for call_ids in schedule.values():
            all_call_ids.extend(call_ids)

        # Sum up the commissions for the calls in this schedule
        for call_id in all_call_ids:
            if call_id in call_commissions:
                commission = call_commissions[call_id]
                try:
                    commission = float(commission)
                except ValueError:
                    commission = 0.0
                total_commission += commission
                processed_call_ids.add(call_id)
            else:
                print(f"Call ID '{call_id}' not found in calls data '{calls_file}'. Assuming zero commission.")

    print(f"\nTotal commission from all calls in all schedules: {total_commission:.2f}")
    print(f"Total number of unique calls processed: {len(processed_call_ids)}")

def main():
    # Paths
    schedules_dir = './Week3/call_schedules'
    calls_dir = './Week3/calls'

    calculate_total_commission(schedules_dir, calls_dir)

if __name__ == '__main__':
    main()
