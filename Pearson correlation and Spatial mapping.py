import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import pearsonr

# --------------------------
# Parameters
# --------------------------
t_min, t_max = 0, 3600          # Time window
pos_thresh, neg_thresh = 0.5, -0.5
dist_thresh = 20                 # Pixel distance threshold for line plotting

mpl.rcParams.update({'font.size': 16})

# --------------------------
# Extract time windowed traces
# --------------------------
time_vector = regions_1.time  # Shared time vector
time_mask = (time_vector >= t_min) & (time_vector <= t_max)

traces_1 = np.stack(regions_1.df["detrended"].apply(lambda x: np.array(x)[time_mask]))
traces_2 = np.stack(regions_2.df["detrended"].apply(lambda x: np.array(x)[time_mask]))
n1, n2 = traces_1.shape[0], traces_2.shape[0]

# ROI coordinates (needed for plotting)
coords_1 = np.array([p[::-1] for p in regions_1.df["peak"].values])
coords_2 = np.array([p[::-1] for p in regions_2.df["peak"].values])

# --------------------------
# Compute correlation matrix
# --------------------------
# Option 1: full pairwise correlation using numpy for speed
correlation_matrix = np.corrcoef(np.vstack([traces_1, traces_2]))
# Extract α-β correlations only
alpha_beta_corr = correlation_matrix[:n1, n1:n1+n2]

# Identify pair types
positive_pairs = np.argwhere(alpha_beta_corr > pos_thresh)
neutral_pairs = np.argwhere((alpha_beta_corr >= -0.3) & (alpha_beta_corr <= 0.3))
negative_pairs = np.argwhere(alpha_beta_corr < neg_thresh)

# --------------------------
# Plot correlation heatmap
# --------------------------
fig, ax = plt.subplots(figsize=(8, 7.2))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0,
            xticklabels=False, yticklabels=False, ax=ax)

# Divider lines between α and β ROIs
ax.axhline(n1, color='black', linestyle='--', linewidth=1)
ax.axvline(n1, color='black', linestyle='--', linewidth=1)

# Labels
ax.text(n1/2, -5, "α-cell ROIs", ha='center', va='center', fontsize=12)
ax.text(n1 + n2/2, -5, "β-cell ROIs", ha='center', va='center', fontsize=12)
ax.text(-10, n1/2, "α-cell ROIs", ha='center', va='center', fontsize=12, rotation=90)
ax.text(-10, n1 + n2/2, "β-cell ROIs", ha='center', va='center', fontsize=12, rotation=90)
ax.set_title("Pairwise Correlation Matrix Between α- and β-cells",  y=1.05)
plt.tight_layout()
plt.show()

# --------------------------
# Plot ROI coordinates & negative correlations
# --------------------------
cmap = plt.cm.coolwarm
norm = mpl.colors.Normalize(vmin=-1, vmax=1)

fig, ax = plt.subplots(figsize=(8, 7.2))

# Plot nodes
ax.scatter(*coords_1.T, color="tab:blue", label="α-cells", alpha=0.9, s=10, zorder=3)
ax.scatter(*coords_2.T, color="tab:orange", label="β-cells", alpha=0.9, s=10, zorder=3)

# Plot negative correlation lines between nearby cells
for i, coord1 in enumerate(coords_1):
    diffs = coords_2 - coord1
    dists = np.linalg.norm(diffs, axis=1)
    mask = (alpha_beta_corr[i] < 0) & (dists < dist_thresh)
    for j in np.where(mask)[0]:
        ax.plot([coord1[0], coords_2[j, 0]],
                [coord1[1], coords_2[j, 1]],
                color=cmap(norm(alpha_beta_corr[i, j])),
                linewidth=1.5, alpha=0.8)

# Formatting
ax.set_title("Negative correlations between nearby α- vs. β-cell pairs", y=1.05)
ax.set_xlabel("X (pixel)")
ax.set_ylabel("Y (pixel)")
ax.invert_yaxis()
ax.grid(True)
ax.legend()

# Colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, cax=cbar_ax, label="Pearson Correlation (r)")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
