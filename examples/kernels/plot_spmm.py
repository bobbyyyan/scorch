import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results from the CSV
results_df = pd.read_csv("spmm_benchmark_results.csv", dtype={"Runtime": float, "NNZ": int})

# Plot settings
sns.set(style="whitegrid")

# Plotting
plt.figure(figsize=(10, 6))
for framework in ['PyTorch', 'Scorch']:
    sub_df = results_df[results_df['Framework'] == framework]
    mean_runtime = sub_df.groupby('NNZ')['Runtime'].mean()
    std_runtime = sub_df.groupby('NNZ')['Runtime'].std()

    plt.plot(mean_runtime.index, mean_runtime, label=f'{framework}')
    plt.fill_between(mean_runtime.index, mean_runtime-std_runtime, mean_runtime+std_runtime, alpha=0.3)

plt.xlabel('Number of Non-Zeros (NNZ)')
plt.ylabel('Average Runtime (seconds)')
plt.title('Sparse Matrix Multiplication Performance')
plt.legend()
plt.xscale('log')
# plt.yscale('log')
plt.savefig("spmm_benchmark_plot.pdf")
plt.show()
