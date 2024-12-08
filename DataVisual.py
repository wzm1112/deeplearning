import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
file_path = 'datasets/generated_time_series_with_anomalies.csv'
data = pd.read_csv(file_path)

# Separate features and anomaly columns
feature_columns = [col for col in data.columns if not col.startswith("Anomaly_edge")]
anomaly_columns = [col for col in data.columns if col.startswith("Anomaly_edge")]

# Calculate correlations
correlation_results = []
for anomaly_col in anomaly_columns:
    for feature_col in feature_columns:
        corr, _ = pearsonr(data[feature_col], data[anomaly_col])
        correlation_results.append({'Feature': feature_col, 'Anomaly': anomaly_col, 'Correlation': corr})

# Create a DataFrame for the correlation results
correlation_df = pd.DataFrame(correlation_results)

# Filter strong correlations
threshold = 0.2  # You can adjust this threshold for more/less stringent filtering
strong_correlations = correlation_df[correlation_df['Correlation'].abs() > threshold]

# Save strong correlations to CSV for further analysis
strong_correlations.to_csv('strong_correlations.csv', index=False)

# Visualize the correlations
def visualize_correlation(data, strong_corrs):
    for _, row in strong_corrs.iterrows():
        feature = row['Feature']
        anomaly = row['Anomaly']
        plt.figure(figsize=(12, 6))
        plt.scatter(data[feature], data[anomaly], alpha=0.5)
        plt.title(f"Correlation Between {feature} and {anomaly} (r={row['Correlation']:.2f})")
        plt.xlabel(feature)
        plt.ylabel(anomaly)
        plt.grid()
        plt.savefig(f"{feature}_vs_{anomaly}.png")


# Visualize the strong correlations
visualize_correlation(data, strong_correlations)

# Display strong correlations
print("Strong Correlations Between Features and Anomalies:")
print(strong_correlations)
