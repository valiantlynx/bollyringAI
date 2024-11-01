import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from ortools.linear_solver import pywraplp
from joblib import Parallel, delayed  # for parallel processing

# Initialize label encoders globally
le_problem = LabelEncoder()
le_difficulty = LabelEncoder()


# Load data from files
def load_data():
    call_reports, calls_data = [], []
    for i in range(10):
        with open(f'./extracted/previous_reports/call_report_{i}.json', 'r') as f:
            call_reports.extend(json.load(f))
        with open(f'./extracted/previous_calls/calls_{i}.json', 'r') as f:
            calls = json.load(f)
            for city, calls_in_city in calls.items():
                for call_id, call_info in calls_in_city.items():
                    call_info.update({"call_id": call_id, "city": city})
                    calls_data.append(call_info)
    call_report_df = pd.DataFrame(call_reports)
    calls_df = pd.DataFrame(calls_data)

    with open('./extracted/workers.json', 'r') as f:
        workers_data = json.load(f)
    workers_df = pd.DataFrame.from_dict(workers_data, orient='index').reset_index().rename(
        columns={'index': 'worker_id'})

    return call_report_df, calls_df, workers_df


# Data Preprocessing
def preprocess_data(call_report_df, calls_df, workers_df):
    merged_df = pd.merge(call_report_df, calls_df, on='call_id')
    merged_df = pd.merge(merged_df, workers_df, left_on='worker_id', right_on='worker_id', how='left')
    merged_df['call_duration_per_profit'] = merged_df['call_time'] / merged_df['call_profit']
    merged_df['technical_problem_encoded'] = le_problem.fit_transform(merged_df['technical_problem'])
    merged_df['difficulty_encoded'] = le_difficulty.fit_transform(merged_df['difficulty'])
    return merged_df


# Model Training
def train_model(merged_df):
    features = ['technical_problem_encoded', 'difficulty_encoded', 'base_salary', 'call_duration_per_profit']
    X = merged_df[features]
    y = merged_df['call_profit']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Parallelized Prediction Function
def predict_worker_performance(model, call_id, call_info, worker):
    features = {
        'technical_problem_encoded': le_problem.transform([call_info['technical_problem']])[0],
        'difficulty_encoded': le_difficulty.transform([call_info['difficulty']])[0],
        'base_salary': worker['base_salary'],
        'call_duration_per_profit': call_info['commission'] / 1  # Placeholder value
    }
    features_df = pd.DataFrame([features])
    predicted_profit = model.predict(features_df)[0]
    return {
        'worker_id': worker['worker_id'],
        'call_id': call_id,
        'predicted_profit': predicted_profit
    }

# Adjusted Prediction for Future Calls
def predict_future_performance(model, future_calls_df, workers_df):
    print("1")
    future_performance = Parallel(n_jobs=-1)(
        delayed(predict_worker_performance)(model, call_id, call_info, worker)
        for city, calls in future_calls_df.items()
        for call_id, call_info in calls.items()
        for _, worker in workers_df.iterrows()
    )
    return pd.DataFrame(future_performance)


# Optimization for Call Schedule
def optimize_schedule(future_performance_df):
    solver = pywraplp.Solver.CreateSolver('SCIP')
    x = {idx: solver.IntVar(0, 1, f'x_{idx}') for idx in future_performance_df.index}
    objective = solver.Objective()
    for idx, row in future_performance_df.iterrows():
        objective.SetCoefficient(x[idx], row['predicted_profit'])
    objective.SetMaximization()
    for call_id in future_performance_df['call_id'].unique():
        solver.Add(
            solver.Sum([x[idx] for idx, row in future_performance_df.iterrows() if row['call_id'] == call_id]) == 1)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("Optimal solution found.")
        return future_performance_df[[x[idx].solution_value() > 0 for idx in range(len(future_performance_df))]]
    else:
        print("No optimal solution found.")
        return None


# Load future calls
def load_future_calls():
    with open('./extracted/feature_calls/calls_11.json', 'r') as f:
        return json.load(f)


# Main Execution

print("kjøerer1")
call_report_df, calls_df, workers_df = load_data()
print("kjøerer2")
merged_df = preprocess_data(call_report_df, calls_df, workers_df)
print("kjøerer3")
model = train_model(merged_df)
print("kjøerer4")
future_calls_df = load_future_calls()
print("kjøerer5")
future_performance_df = predict_future_performance(model, future_calls_df, workers_df)
print("kjøerer6")
optimized_schedule = optimize_schedule(future_performance_df)
print("kjøerer7")


# Save schedule to JSON
def save_schedule(optimized_schedule):
    if optimized_schedule is not None:
        schedule = {}
        for _, row in optimized_schedule.iterrows():
            schedule.setdefault(row['worker_id'], []).append(row['call_id'])
        with open('optimized_call_schedule.json', 'w') as f:
            json.dump(schedule, f)
        print("Schedule saved to 'optimized_call_schedule.json'.")


save_schedule(optimized_schedule)
