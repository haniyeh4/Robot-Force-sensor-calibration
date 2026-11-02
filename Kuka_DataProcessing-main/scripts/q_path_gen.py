import csv

# === Fixed DOF Values ===
dof1 = 0.0
dof2 = -60.0
dof3 = 90.0
dof6 = 0.0

# === CSV Output File ===
OUTPUT_FILE = 'data/paths/1_TheRightWay.csv'

# === Initialize the CSV file ===
with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['DOF1', 'DOF2', 'DOF3', 'DOF4', 'DOF5', 'DOF6'])  # header

# === Update Function ===
def update(dof4, dof5):
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dof1, dof2, dof3, dof4, dof5, dof6])

# === Step Size ===
step = 5  # degrees

# === Ranges for DOF4 and DOF5 ===
range_dof4 = list(range(-45, 46, step))     # -45 to +45 inclusive
range_dof5 = list(range(15, 106, step))     # 15 to 105 inclusive

# === Zigzag Sweep ===
for i, dof4 in enumerate(range_dof4):
    if i % 2 == 0:
        iter_d5 = range_dof5
    else:
        iter_d5 = reversed(range_dof5)

    for dof5 in iter_d5:
        update(dof4, dof5)
