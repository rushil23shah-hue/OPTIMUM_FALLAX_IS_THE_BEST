import pandas as pd
import matplotlib.pyplot as plt
import os

df = pd.read_csv("results/results.csv")

os.makedirs("results/plots", exist_ok=True)

metrics = [
    "Reward",
    "Distance",
    "Velocity",
    "Angle",
    "COM Jitter",
    "Episode Length"
]

for metric in metrics:

    plt.figure(figsize=(8,5))

    plt.bar(df["Model"], df[metric])

    plt.title(metric)
    plt.xlabel("Reward Function")
    plt.ylabel(metric)

    plt.xticks(rotation=20)

    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(f"results/plots/{metric.replace(' ','_')}.png", dpi=300)

    plt.close()

print("Plots saved successfully!")