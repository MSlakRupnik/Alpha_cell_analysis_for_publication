import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Example ROI lists
alpha = [100, ...]
delta = [200, ...]

# Function: median active fraction per minute
def compute_median_active(events_df, roi_list):
    df = events_df[events_df.roi.isin(roi_list)].copy()
    df = df[(df["t0"] >= 0) & (df["t0"] <= 3600)].copy()
    df["minute_bin"] = np.floor(df["t0"] / 60).astype(int)
    active_time_df = df.groupby(["roi", "minute_bin"])["halfwidth"].sum().reset_index()
    active_time_df.rename(columns={"halfwidth": "active_time"}, inplace=True)
    active_time_df["active_fraction"] = active_time_df["active_time"] / 60.0
    median_active = active_time_df.groupby("minute_bin")["active_fraction"].median().reset_index()
    median_active["time_center"] = median_active["minute_bin"] * 60 + 30
    return median_active

# Function: median of medians per range
def compute_median_active_ranges(events_df, roi_list, ranges):
    df = events_df[events_df.roi.isin(roi_list)].copy()
    df = df[(df["t0"] >= 0) & (df["t0"] <= 3600)].copy()
    median_list = []
    for start, end in ranges:
        df_range = df[(df["t0"] >= start) & (df["t0"] < end)].copy()
        df_range["minute_bin"] = np.floor(df_range["t0"] / 60).astype(int)
        active_time_df = df_range.groupby(["roi", "minute_bin"])["halfwidth"].sum().reset_index()
        active_time_df.rename(columns={"halfwidth": "active_time"}, inplace=True)
        active_time_df["active_fraction"] = active_time_df["active_time"] / 60.0
        median_val = active_time_df.groupby("minute_bin")["active_fraction"].median().median()
        median_list.append(median_val)
    return median_list

# Define time ranges
time_ranges = [(0,600), (600,1200), (1200,1800), (1800,2400), (2400,3000), (3000,3600)]

# Compute granular per-minute medians
alpha_active = compute_median_active(Events, alpha)
delta_active = compute_median_active(Events, delta)

# Compute median-of-median per range
alpha_medians = compute_median_active_ranges(Events, alpha, time_ranges)
delta_medians = compute_median_active_ranges(Events, delta, time_ranges)
x_pos = [start + (end-start)/2 for start, end in time_ranges]

# Plot
plt.figure(figsize=(10,4))

# Scatter plot (per-minute medians)
plt.scatter(alpha_active["time_center"], alpha_active["active_fraction"],
            s=40, color="red", marker="o", alpha=0.6, label="α-cell ROI100")
plt.scatter(delta_active["time_center"], delta_active["active_fraction"],
            s=40, color="green", marker="^", alpha=0.6, label="δ-cell ROI200")

# Line plot (median of medians)
plt.plot(x_pos, alpha_medians, color="darkred", marker="o", linewidth=2)
plt.plot(x_pos, delta_medians, color="darkgreen", marker="^", linewidth=2)

# Vertical lines for ranges
for _, end in time_ranges:
    plt.axvline(end, color="lightgray", linestyle="--")

# Glucose concentration annotations
glucose_concentrations = [1.8, 3.6, 5.4, 7.2, 9.0, 10.8]
for i, (start, end) in enumerate(time_ranges):
    plt.text((start+end)/2, 1.22, f"{glucose_concentrations[i]} mM",
             ha='center', fontsize=14)

# Axis labels & legend
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel("Active Fraction (Active time/60s)", fontsize=14)
plt.title("Figure 1", fontsize=14, x=0, pad=30,fontweight='bold')
plt.xlim(0, 3600)
plt.ylim(0, 1.2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.show()
