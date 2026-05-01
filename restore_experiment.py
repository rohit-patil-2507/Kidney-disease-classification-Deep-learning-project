import os
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/rohit-patil-2507/Kidney-disease-classification-Deep-learning-project.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'rohit-patil-2507'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'eceeb5ddbc868faee671550f4c23909cdd1b7e29'

import mlflow
from mlflow.tracking import MlflowClient

experiment_name = 'Kidney-Disease-Classification'

# Get experiment by name - this returns both active and deleted experiments
exp = mlflow.get_experiment_by_name(experiment_name)

print(f'Experiment found: {exp.name}')
print(f'  ID: {exp.experiment_id}')
print(f'  Lifecycle: {exp.lifecycle_stage}')

if exp.lifecycle_stage == 'deleted':
    print(f'\nRestoring experiment {exp.experiment_id}...')
    client = MlflowClient()
    client.restore_experiment(exp.experiment_id)
    print('Restored successfully!')
    
    # Verify restoration
    restored_exp = mlflow.get_experiment_by_name(experiment_name)
    print(f'\nAfter restoration:')
    print(f'  Lifecycle: {restored_exp.lifecycle_stage}')
else:
    print('\nExperiment is already active!')
