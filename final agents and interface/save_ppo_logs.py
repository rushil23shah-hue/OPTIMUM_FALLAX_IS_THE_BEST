import csv
import re
import matplotlib.pyplot as plt

# Paste your copied terminal text between the triple quotes
terminal_output = """
  step    20000 | eval_mean_return =   -23.48
  step    40000 | eval_mean_return =   -25.25
  step    60000 | eval_mean_return =   -28.00
  step    80000 | eval_mean_return =   -87.01
  step   100000 | eval_mean_return =   -64.51
  step   120000 | eval_mean_return =   -97.10
  step   140000 | eval_mean_return =    45.17
  step   160000 | eval_mean_return =    -9.27
  step   180000 | eval_mean_return =   113.66
  step   200000 | eval_mean_return =   132.19
  step   220000 | eval_mean_return =   154.53
  step   240000 | eval_mean_return =    89.51
  step   260000 | eval_mean_return =   169.60
  step   280000 | eval_mean_return =   181.62
  step   300000 | eval_mean_return =   180.49
  step   320000 | eval_mean_return =   184.13
  step   340000 | eval_mean_return =   100.05
  step   360000 | eval_mean_return =   195.07
  step   380000 | eval_mean_return =   170.59
  step   400000 | eval_mean_return =   195.77
  step   420000 | eval_mean_return =   199.64
  step   440000 | eval_mean_return =   125.32
  step   460000 | eval_mean_return =   115.95
  step   480000 | eval_mean_return =    94.40
  step   500000 | eval_mean_return =    98.22
  step   520000 | eval_mean_return =   148.73
  step   540000 | eval_mean_return =   144.70
  step   560000 | eval_mean_return =   190.40
  step   580000 | eval_mean_return =   126.76
  step   600000 | eval_mean_return =   154.64
  step   620000 | eval_mean_return =   144.81
  step   640000 | eval_mean_return =   148.33
  step   660000 | eval_mean_return =   160.39
  step   680000 | eval_mean_return =   135.55
  step   700000 | eval_mean_return =   212.04
  step   720000 | eval_mean_return =   169.79
  step   740000 | eval_mean_return =   225.11
  step   760000 | eval_mean_return =   221.87
  step   780000 | eval_mean_return =   181.90
  step   800000 | eval_mean_return =   224.06
  step   820000 | eval_mean_return =   223.47
  step   840000 | eval_mean_return =   223.20
  step   860000 | eval_mean_return =   223.54
  step   880000 | eval_mean_return =   224.63
  step   900000 | eval_mean_return =   221.96
  step   920000 | eval_mean_return =   228.80
  step   940000 | eval_mean_return =   214.23
  step   960000 | eval_mean_return =   221.42
  step   980000 | eval_mean_return =   149.22
  step  1000000 | eval_mean_return =   208.97
  step  1020000 | eval_mean_return =   167.30
  step  1040000 | eval_mean_return =   169.28
  step  1060000 | eval_mean_return =   231.93
  step  1080000 | eval_mean_return =   222.42
  step  1100000 | eval_mean_return =   229.04
  step  1120000 | eval_mean_return =   226.27
  step  1140000 | eval_mean_return =   178.30
  step  1160000 | eval_mean_return =   218.77
  step  1180000 | eval_mean_return =   176.64
  step  1200000 | eval_mean_return =   212.57
  step  1220000 | eval_mean_return =   189.19
"""

# Extract steps and evaluation returns
pattern = r"step\s+(\d+)\s+\|\s+eval_mean_return\s+=\s+([-+]?\d*\.\d+|\d+)"
matches = re.findall(pattern, terminal_output)

steps = [int(m[0]) for m in matches]
returns = [float(m[1]) for m in matches]

# Save CSV
with open("ppo_simple_rewards.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Step", "Eval_Mean_Return"])
    for s, r in zip(steps, returns):
        writer.writerow([s, r])

# Save PNG Plot
plt.figure(figsize=(10, 5))
plt.plot(steps, returns, label="Eval Mean Return", color="orange", linewidth=2)
plt.xlabel("Total Steps")
plt.ylabel("Eval Mean Return")
plt.title("Standard PPO Training Curve")
plt.grid(True)
plt.legend()
plt.savefig("ppo_simple_curve.png", dpi=300)
plt.show()

print("Successfully saved ppo_simple_rewards.csv and ppo_simple_curve.png!")