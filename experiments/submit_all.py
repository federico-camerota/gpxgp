import os

for f in os.listdir('../jobs'):
    if f != 'run.sh':
        os.system(f'sbatch {os.path.join("jobs", f)}')
