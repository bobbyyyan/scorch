import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_filename = "spmspm_benchmark_results.csv"

# Load results from the CSV
results_df = pd.read_csv(csv_filename, dtype={"Runtime": float, "NNZ": int})

# Plot settings
sns.set(style="white", context="talk")

# Plotting
plt.figure(figsize=(15, 6))

plt.rcParams.update(
    {
        "grid.linestyle": " ",
        "font.size": 17,
        "axes.labelsize": 22,
        "axes.titlesize": 24,
        "xtick.labelsize": 22,
        "ytick.labelsize": 18,
        # Legend settings
        "legend.fontsize": 22,
        "legend.title_fontsize": 24,
        # Legend dot size
        "legend.markerscale": 5,
    }
)

colors = ["#19526c", "#fc764a", "#1AACAC", "#E7B10A", "#ED5AB3"]

framework_colors = {
    "PyTorch": colors[0],
    "Scorch": colors[1],
}

for framework in ["Scorch", "PyTorch"]:
    sub_df = results_df[results_df["Framework"] == framework]
    mean_runtime = sub_df.groupby("NNZ")["Runtime"].mean()
    std_runtime = sub_df.groupby("NNZ")["Runtime"].std()

    # Using scatter plot for data points, adjust size using 's' and transparency using 'alpha'
    plt.scatter(
        mean_runtime.index,
        mean_runtime,
        label=f"{framework}",
        s=2,
        alpha=0.6,
        color=framework_colors[framework],
    )  # smaller size
    # Optional: Adjust error bars
    # plt.errorbar(mean_runtime.index, mean_runtime, yerr=std_runtime, fmt='o', alpha=0.3, capsize=3, markersize=2)

# plt.xlabel("Number of Non-Zeros (NNZ)")
# plt.ylabel("Average Runtime (seconds)")
plt.title("SpMSpM Runtime")
# plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("spmspm.pdf", bbox_inches="tight")
plt.savefig("spmspm.svg", bbox_inches="tight")
plt.show()
