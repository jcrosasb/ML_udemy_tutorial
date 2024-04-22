import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/data_reg.csv').sample(n=100, random_state=42)

# Select the column you want to plot against others
selected_column = 'Temperature_C'

# Determine the number of rows and columns for subplots
num_cols = 2  # Number of columns of subplots
num_rows = (df.shape[1] - 1) // num_cols + 1  # Number of rows of subplots

# Create subplots
fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(12, 8))

# Flatten the axes if there's only one row or column
if num_rows == 1:
    axes = axes.reshape(1, -1)
if num_cols == 1:
    axes = axes.reshape(-1, 1)

# Iterate through each column and plot against the selected column
for i, column in enumerate(df.columns):
    if column != selected_column:
        row_index = i // num_cols
        col_index = i % num_cols
        ax = axes[row_index, col_index]
        ax.scatter(df[selected_column], df[column])
        ax.set_xlabel(selected_column)
        ax.set_ylabel(column)
        ax.set_title(f"{column} vs {selected_column}")

# Hide any unused subplots
for i in range(len(df.columns), num_rows * num_cols):
    fig.delaxes(axes.flatten()[i])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


