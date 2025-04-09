# Chapter 5
# Experiment 2 data preparation
# This data preparation file produces df_strategy_....csv files and summary_states_df_strategy files.

# Load libraries
library(tidyverse)

# Define the path and pattern
path <- "exp2-results/"
path <- "exp3-results/" # alternatively
pattern <- ".*-1-regagents-trust-data\\.csv" # # change number based on strategy id

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
write.csv(stats_df, "exp2-results/attack-strategies/summary_stats_df_strategy_1.csv",
          row.names=FALSE)
write.csv(stats_df, "exp3-results/attack-strategies/summary_stats_df_strategy_1.csv",
          row.names=FALSE)

#####
# The following code is necessary for data visualisation
#####
# Define the pattern for the files
setwd("exp2-results")
file_pattern <- "*.csv"

# Get the list of files matching the patterns
files <- list.files(pattern = file_pattern)

# Function to add step column
add_step_column <- function(file_name) {
  # Read the CSV file
  data <- read.csv(file_name)
  
  # Add the step column starting from 0
  data$step <- seq(0, nrow(data) - 1)
  
  # Write the modified data back to the file
  write.csv(data, file_name, row.names = FALSE)
}

# Iterate over the list of files and apply the function
for (file in files) {
  if (grepl("attack-order.csv$", file) | grepl("attack-actions-count.csv$", file)) {
    add_step_column(file)
  }
}

# Prepare data based on attack strategies before resilience evaluation
# Attack strategies
order_files <- list.files(path, pattern = "*attack-order.csv", full.names = TRUE)
count_files <- list.files(path, pattern = "*attack-actions-count.csv", full.names = TRUE)

# If necessary, add initial seed value over all documents to indicate simulation id
process_order_file <- function(file) {
  # Extract the simulation_id (seed value) from the filename
  strategy_id <- gsub(".*-([0-9]+)-attack-order\\.csv$", "\\1", basename(file))
  # Read the CSV file
  df <- read.csv(file)
  # Add the simulation_id as the first column
  df <- cbind(strategy_id = as.numeric(strategy_id), df)
  return(df)
}

process_count_file <- function(file) {
  # Extract the simulation_id (seed value) from the filename
  strategy_id <- gsub(".*-([0-9]+)-attack-actions-count\\.csv$", "\\1", basename(file))
  # Read the CSV file
  df <- read.csv(file)
  # Add the simulation_id as the first column
  df <- cbind(strategy_id = as.numeric(strategy_id), df)
  return(df)
}

# Apply the function to each file and store dataframes in a list
order_dfs <- lapply(order_files, process_order_file)

# Apply the function to each file and store dataframes in a list
count_dfs <- lapply(count_files, process_count_file)

# Remove the first row from each dataframe in order_dfs
order_dfs <- lapply(order_dfs, function(df) df[-1, ])

# Remove the first row from each dataframe in count_dfs
count_dfs <- lapply(count_dfs, function(df) df[-1, ])

# Optionally combine all dataframes into one
#combined_order_df <- do.call(rbind, processed_order_dfs)
combined_order_df <- do.call(rbind, order_dfs)
combined_order_df <- na.omit(combined_order_df)

# Split the dataframe into a list of dataframes based on 'strategy_id'
strategy_dfs <- split(combined_order_df, combined_order_df$strategy_id)

# Get time in attack stages for each simulation_id, step
# This function merges a single order_strategy dataframe with all count_dfs
merge_with_count_dfs <- function(strategy_dfs) {
  merged_df <- NULL
  for (count_df in count_dfs) {
    # Merge based on simulation_id and step
    print(count_df)
    temp_merged <- merge(strategy_dfs, count_df, by = c("strategy_id","step"))
    
    # If merged_df is not yet initialized, initialize it with temp_merged
    if (is.null(merged_df)) {
      merged_df <- temp_merged
    } else {
      # Else, bind the rows of temp_merged to merged_df
      merged_df <- rbind(merged_df, temp_merged)
    }
  }
  return(merged_df)
}

# Apply the function to each order_strategy dataframe
final_dfs <- lapply(strategy_dfs, merge_with_count_dfs)

# List of column names to remove
columns_to_remove <- c("recon.x", "cyber.x", "disinfo.x", 
                       "comb.x", "cyber2.x", "disinfo2.x")

# Function to remove specified columns from a dataframe
remove_columns <- function(df) {
  df[, !(names(df) %in% columns_to_remove)]
}

# Apply this function to each dataframe in final_dfs
final_dfs_cleaned <- lapply(final_dfs, remove_columns)

# Function to process each dataframe
process_df <- function(df) {
  # Create a unique key for each combination of strategy_id and attack stages
  df$unique_key <- apply(df[, c("strategy_id", "recon.y", "cyber.y", 
                                "disinfo.y", "comb.y", 
                                "cyber2.y", "disinfo2.y")], 
                         1, paste, collapse = "-")
  
  # Split the dataframe based on this unique key
  split_dfs <- split(df, df$unique_key)
  
  # Remove the 'unique_key' column from each dataframe
  split_dfs <- lapply(split_dfs, function(x) x[, !(names(x) %in% "unique_key")])
  
  return(split_dfs)
}

# Apply this function to each dataframe in final_dfs_cleaned
separated_dfs <- lapply(final_dfs_cleaned, process_df)

# Define the row threshold to only keep more popular attack strategies 
row_threshold <- 0
step_threshold <- 0

# Initialize an index for naming
df_index <- 1

# Dictionary to store the most popular variant of each strategy
strategy_counts <- list()

# Iterate over each list in separated_dfs
for (strategy_id in seq_along(separated_dfs)) {
  # Get the list for this strategy_id
  strategy_list <- separated_dfs[[strategy_id]]
  
  # Temporary storage for strategies that pass the threshold
  temp_storage <- list()
  
  # Iterate over each dataframe in the strategy list
  for (df_key in names(strategy_list)) {
    # Get the dataframe
    df <- strategy_list[[df_key]]
    
    # Calculate the number of rows where step value is greater than the step threshold
    num_rows_with_step_gt_step_threshold <- sum(df$step > step_threshold)
    
    # Check if the dataframe has more than the row threshold rows and all these rows have step value greater than the step threshold
    if (nrow(df) > row_threshold && num_rows_with_step_gt_step_threshold == nrow(df)) {
      # Store the dataframe in temporary storage
      temp_storage[[df_key]] <- df
    }
  }
  
  # If there are valid strategies in temp_storage
  if (length(temp_storage) > 0) {
    # Determine the most popular strategy variant
    strategy_count <- sapply(temp_storage, nrow)
    most_popular_key <- names(temp_storage)[which.max(strategy_count)]
    most_popular_df <- temp_storage[[most_popular_key]]
    
    # Use strategy_id directly
    real_strategy_id <- as.character(strategy_id)
    
    # Check if this strategy is already in the dictionary
    if (!is.null(strategy_counts[[real_strategy_id]])) {
      # Compare with the stored variant, keep the one with the most rows
      if (nrow(most_popular_df) > strategy_counts[[real_strategy_id]]$count) {
        strategy_counts[[real_strategy_id]]$df <- most_popular_df
        strategy_counts[[real_strategy_id]]$count <- nrow(most_popular_df)
      }
    } else {
      # Store this new strategy variant
      strategy_counts[[real_strategy_id]] <- list(df = most_popular_df, count = nrow(most_popular_df))
    }
  }
}

# Assign each strategy dataframe to a unique name in the global environment
for (real_strategy_id in names(strategy_counts)) {
  df_name <- paste0("df_strategy_", real_strategy_id)
  assign(df_name, strategy_counts[[real_strategy_id]]$df, envir = .GlobalEnv)
}

# Write all strategies to csv
# Get a list of all dataframes that start with 'df_strategy_'
df_list <- ls(pattern = "^df_strategy_")

# Define the base path where you want to save the files
base_path <- "exp2-results/attack-strategies/"
# Iterate through each dataframe name in the list
for (df_name in df_list) {
  # Get the actual dataframe object by name
  dat <- get(df_name)
  
  # Create the full file path, assuming df_name is the name you want to use for the file
  file_path <- paste0(base_path, df_name, ".csv")
  
  # Save the dataframe to CSV
  write.csv(dat, file_path, row.names = FALSE)
}
