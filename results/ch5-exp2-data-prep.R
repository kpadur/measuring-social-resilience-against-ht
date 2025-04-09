# Chapter 5
# Experiment 2 data preparation
# This data preparation file produces df_strategy_....csv files and summary_states_df_strategy files.

# Load libraries
library(tidyverse)

# Define the path and pattern
path <- "/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2-results/"
pattern <- "*-5-regagents-trust-data\\.csv"

# List and read all matching files
files <- list.files(path = path, pattern = pattern, full.names = TRUE)
combined_df <- files %>%
  lapply(read_csv) %>%
  bind_rows()

# Separate the combined dataframe by episode number
# Replace 'episode' with the actual column name
dataframes_by_episode <- split(combined_df, combined_df$episode)

# Combine data from all episodes into one dataframe
combined_all_episodes <- bind_rows(dataframes_by_episode)

# Function to compute mean and standard deviation for each time step
compute_stats <- function(df) {
  df %>%
    group_by(timestep) %>%
    summarise(across(everything(), list(mean = ~mean(.), sd = ~sd(.)), .names = "{.col}_{.fn}")) %>%
    ungroup()
}

# Ensure 'timestep' column exists
if ("timestep" %in% colnames(combined_all_episodes)) {
  # Compute statistics for the combined dataframe
  stats_df <- compute_stats(combined_all_episodes)
  
  # Optionally, rename columns
  colnames(stats_df) <- gsub("_mean", "_mean", colnames(stats_df))
  colnames(stats_df) <- gsub("_sd", "_sd", colnames(stats_df))
  
  # View the final results
  print(stats_df)
} else {
  stop("Column 'timestep' not found in the combined dataframe.")
}

plot(subset(stats_df$group3_trust_in_sp1_mean, stats_df$timestep > 0))

# Write summary_stats to csv
write.csv(stats_df, "/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2-results/attack-strategies/summary_stats_df_strategy_5.csv",
          row.names=FALSE)
