import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Load the dataset
df = pd.read_csv(sys.argv[1])

# Extract 'time' and 'weight' from the 'Retriever' column
df[['time', 'weight']] = df['Retriever'].str.extract(r'time=([^,]+), weight=([^\)]+)')

# Convert weight to numeric
df['weight'] = pd.to_numeric(df['weight'])

# Set the style for the plots
sns.set(style="whitegrid")

# Create a figure for the plots
plt.figure(figsize=(20, 15))

# Plot 1: Boxplot of MAP scores by Expansion Method


# Plot 2: Line plot of MAP vs. weight for each expansion method, separated by time
# Get unique time values
time_values = df['time'].unique()

# Create subplots for each time value
for i, time_val in enumerate(time_values):
    plt.subplot(2, 2, i + 1)
    subset_df = df[df['time'] == time_val]
    sns.lineplot(x='weight', y='MAP', hue='Expansion_Method', data=subset_df, marker='o')
    plt.title(f'MAP vs. Weight (time={time_val})')
    plt.xlabel('Weight')
    plt.ylabel('MAP Score')
    plt.legend(title='Expansion Method', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.suptitle('Analysis of Expansion Methods for Information Retrieval', fontsize=24, y=1.03)
plt.savefig("expansion_method_analysis.png")
plt.show()


# Bar chart to compare the best MAP for each method
# Find the best MAP score for each Expansion_Method
best_map = df.loc[df.groupby('Expansion_Method')['MAP'].idxmax()]
best_map = best_map.sort_values(by='MAP', ascending=False)

plt.figure(figsize=(12, 7))
ax = sns.barplot(x='Expansion_Method', y='MAP', data=best_map)
plt.title('Best MAP Score for Each Expansion Method')
plt.xlabel('Expansion Method')
plt.ylabel('Best MAP Score')
plt.xticks(rotation=45, ha='right')

# Add the MAP values on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.4f'),
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'center',
                   xytext = (0, 9),
                   textcoords = 'offset points')

plt.tight_layout()
plt.savefig("best_map_scores.png")
plt.show()