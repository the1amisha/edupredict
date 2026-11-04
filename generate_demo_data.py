"""Create reproducible, fully synthetic student data for classroom UI testing.

The final_result column is simulated, not real exam outcome evidence.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'demo_students.csv'
rng = np.random.default_rng(42)

first_names = [
    'Aarav', 'Priya', 'Rohan', 'Sneha', 'Kabir', 'Ananya', 'Meera', 'Ishaan',
    'Nisha', 'Dev', 'Tanya', 'Arjun', 'Riya', 'Aditya', 'Simran', 'Kunal',
    'Pooja', 'Varun', 'Diya', 'Manav', 'Sana', 'Neha', 'Vivaan', 'Kriti',
]
last_names = ['Sharma', 'Verma', 'Kumar', 'Singh', 'Das', 'Sinha', 'Gupta', 'Roy', 'Jha', 'Mehta', 'Patel', 'Bose']

n = 280
attendance = np.clip(rng.normal(72, 17, n), 25, 100)
ability = rng.normal(0, 1, n)
assignment = np.clip(64 + 12*ability + 0.22*(attendance-70) + rng.normal(0, 13, n), 5, 100)
internal = np.clip(61 + 13*ability + 0.23*(attendance-70) + rng.normal(0, 13, n), 5, 100)
# Teaching-only outcome generation: hidden and probabilistic; not a real examination result.
pass_logit = (0.26*(attendance-64) + 0.32*(assignment-58) + 0.42*(internal-57)) / 12
pass_probability = 1/(1+np.exp(-pass_logit))
failed = rng.binomial(1, 1-pass_probability)
rows = []
for i in range(n):
    rows.append({
        'student_id': f'ST-{i+1:04d}',
        'name': f'{first_names[i % len(first_names)]} {last_names[(i//len(first_names)) % len(last_names)]}',
        'attendance': int(round(attendance[i])),
        'assignment': int(round(assignment[i])),
        'internal': int(round(internal[i])),
        'final_result': 'Fail' if failed[i] else 'Pass'
    })
OUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f'Created {n} synthetic student records: {OUT}')
