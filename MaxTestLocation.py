import json
import os
from collections import Counter


def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def get_worker_locations(call_reports_path, calls_path):
    worker_locations = {}

    # Load all historical calls data into a dictionary for quick lookup
    calls_data = {}
    for calls_file in os.listdir(calls_path):
        if calls_file.endswith('.json'):
            calls_file_path = os.path.join(calls_path, calls_file)
            calls_json = load_json(calls_file_path)
            for location, calls in calls_json.items():
                for call_id in calls:
                    calls_data[call_id] = location

    # Process call reports to map workers to locations
    for report_file in os.listdir(call_reports_path):
        if report_file.endswith('.json'):
            report_path = os.path.join(call_reports_path, report_file)
            report_data = load_json(report_path)
            for call in report_data:
                worker_id = call["worker_id"]
                call_id = call["call_id"]
                location = calls_data.get(call_id)
                if location:
                    if worker_id not in worker_locations:
                        worker_locations[worker_id] = []
                    worker_locations[worker_id].append(location)

    # Determine the most common location for each worker
    worker_primary_locations = {}
    for worker_id, locations in worker_locations.items():
        most_common_location = Counter(locations).most_common(1)[0][0]
        worker_primary_locations[worker_id] = most_common_location

    return worker_primary_locations


def check_call_assignments(schedule_file, calls_file, worker_locations, output_file):
    schedule_data = load_json(schedule_file)
    calls_data = load_json(calls_file)

    # Build a mapping of call_id to location for the future calls
    future_calls_locations = {}
    for location, calls in calls_data.items():
        for call_id in calls:
            future_calls_locations[call_id] = location

    total_calls = 0
    matched_calls = 0
    output = []

    for worker_id, assigned_calls in schedule_data.items():
        worker_location = worker_locations.get(worker_id, "Unknown")
        output.append(f"Worker {worker_id}: {worker_location}")

        for call_id in assigned_calls:
            call_location = future_calls_locations.get(call_id, "Unknown")
            output.append(f"  Call {call_id}: {call_location}")
            total_calls += 1
            if call_location == worker_location:
                matched_calls += 1

        output.append("----------------------------")

    accuracy = (matched_calls / total_calls) * 100 if total_calls else 0
    output.append(f"{accuracy:.2f}% of calls are correctly assigned by location.\n")

    # Write output to a file
    with open(output_file, 'w') as f:
        for line in output:
            f.write(line + '\n')


# Define paths to your JSON files
call_reports_path = 'extracted/previous_reports'  # e.g., 'extracted/previous_reports'
calls_path = 'Week3/calls'  # e.g., 'extracted/previous_calls'
schedule_file = 'Week3/call_schedules/call_schedule_calls_32.json'  # e.g., 'extracted/new_schedule.json'
calls_file = 'Week3/calls/calls_32.json'  # e.g., 'extracted/future_calls/calls_11.json'
output_file = 'output_results.txt'  # Path for the output file

# Load worker locations based on historical data
worker_locations = get_worker_locations(call_reports_path, calls_path)

# Check call assignments in the new schedule and save output to file
check_call_assignments(schedule_file, calls_file, worker_locations, output_file)