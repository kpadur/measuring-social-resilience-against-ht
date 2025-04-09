# Chapter 5
# Experiment 2
# Import libraries
library(tidyverse)
library(pracma)

library(psych)
library(dplyr)
library(tidyr)
library(dplyr)
library(ggplot2)
library(plotly)

library(reshape2)
library(stringr)

library(mcp)


# Resilience analysis
#####
# Piece-wise linear regression analysis
#####
# Use exp1 data to represent 'real' (forcasted) trust values over time
targeted_trust <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp1-results/summary_regagents_trust.csv",
                           header = TRUE)
# Use exp2 data to represent 'observed' trust values over time
summary_stats <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2-results/attack_strategies/summary_stats_df_strategy_1.csv",
                          header = TRUE)


# Determine change points in data
warm_up_period <- 0
end <- 500
group_df <- summary_stats %>% select(timestep, group0_trust_in_sp1_mean)
plot(group_df)

# Disruption time
t_DE <- 202

# Define the model
model <- list(
  mean_trust_in_sp1 ~ 1, # prevention phase: flat before t_D
  ~ timestep, # absorption phase : changes after t_DP
  ~ 1,
  ~ timestep, # recovery phase: changes after t_LP
  ~1 # adaptation phase: flat after t_RP
)

set.seed(123)
# Fit the model
fit <- mcp(model, data = group_df, par_x = "timestep",
           prior = list(
             cp_1 = paste0("dnorm(", 201, ", 1e-6)"),
             cp_2 = paste0("dnorm(", 202,", 160)"),
             cp_3 = paste0("dnorm(", 160, ", 300)"),
             cp_4 = paste0("dnorm(", 300, ", 350)")
           ),
           iter = 50000)
# View summary
summary(fit)
plot(fit) + plot_pars(fit, pars = c("cp_1", "cp_2", "cp_3"), type = "dens_overlay")

# Determine disruption point, landing point, and recovery point
summary_fit <- summary(fit)

t_DP <- round(summary_fit$mean[summary_fit$name == "cp_1"])
t_LP <- round(summary_fit$mean[summary_fit$name == "cp_2"])
t_AP <- round(summary_fit$mean[summary_fit$name == "cp_3"])
t_RP <- round(summary_fit$mean[summary_fit$name == "cp_3"])
t_RP <- round(summary_fit$mean[summary_fit$name == "cp_4"])
# Record data in csv
print(t_DP)
print(t_LP)
print(t_AP)
print(t_RP)

# #####
# Visual inspection and metrics calculation
#####
# Analyse specific group's resilience - NOTE: Pick one of the following 2 lines
setwd("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/")
targeted_trust <- read.csv("exp1-results/summary_regagents_trust.csv", header = TRUE)
# Find max and min trust values specifically for Provider 2 for every group (not including time step 0)
summary_stats <- read.csv("exp2-results/attack-strategies/summary_stats_df_strategy_5.csv",header = TRUE)
# Analyse specific group's resilience
group_df <- summary_stats %>% select(timestep, group0_trust_in_sp1_mean)
colnames(group_df)[c(1,2)] <- c("timestep", "mean_trust_in_sp1")
target_trust_sp1_df <- targeted_trust %>% select(timestep, group0_trust_in_sp1_mean)
colnames(target_trust_sp1_df)[c(1,2)] <- c("timestep", "mean_trust_in_sp1")

# modify
target_trust_sp1_df <- target_trust_sp1_df %>% 
  select(timestep, mean_trust_in_sp1) %>% 
  mutate(mean_trust_in_sp1 = mean_trust_in_sp1 + 0.0025) # 2 0.0025
target_trust_sp1_df <- target_trust_sp1_df %>% 
  select(timestep, mean_trust_in_sp1) %>% 
  mutate(mean_trust_in_sp1 = mean_trust_in_sp1 - 0.001) # 0

#####
# Plot the data
plot(group_df$timestep, group_df$mean_trust_in_sp1, xaxt = 'n', xlab = "timestep", ylab = "mean_trust_in_sp1")
# Customize the x-axis with specified intervals
axis(1, at = seq(100, max(group_df$timestep), by = 10))


# Plot the first scatter plot
# Filter out the first 100 timesteps
group_filtered <- subset(group_df, timestep > 100)
target_filtered <- subset(target_trust_sp1_df, timestep > 100)

# Plot the first dataset
plot(group_filtered$timestep, group_filtered$mean_trust_in_sp1, col = "red", pch = 16,
     xlab = "timestep", ylab = "mean_trust_in_sp1", main = "Comparison of Trust in sp1")

# Add the second dataset
points(target_filtered$timestep, target_filtered$mean_trust_in_sp1, col = "blue", pch = 16)

# Customize the x-axis with specified intervals
axis(1, at = seq(110, max(group_filtered$timestep, target_filtered$timestep), by = 10))

# Add a legend
legend("topright", legend = c("group_df", "target_trust_sp1_df"), 
       col = c("red", "blue"), pch = 16)


# Subset the data for timesteps between 200 and 300
subset_df <- group_df[group_df$timestep >= 200 & group_df$timestep <= 300, ]
# Find the index of the minimum mean_trust_in_sp1
min_index <- which.min(subset_df$mean_trust_in_sp1)
# Get the corresponding timestep and minimum value
min_timestep <- subset_df$timestep[min_index]
min_value <- subset_df$mean_trust_in_sp1[min_index]
# Print results
cat("Minimum mean_trust_in_sp1:", min_value, 
    "at timestep:", min_timestep, "\n")

#####
# Resilience loss calculated manually
#####
t_DP <- 202
t_RP <- 310
  # Filter data about phases
real_absorption_recovery_phases_df <- subset(group_df, timestep >= t_DP & timestep < t_RP)
target_absorption_recovery_phases_df <- subset(target_trust_sp1_df, timestep >= t_DP & timestep < t_RP)
  
# Cumulative impact as 'loss of resilience' (difference between the target and real trust)
#EQ1: \int_{t_DP}^{t_{RP}} [S_T(t) - S_R(t)] dt
real_absorption_recovery_phases_df$difference <-  target_absorption_recovery_phases_df$mean_trust_in_sp1 -
    real_absorption_recovery_phases_df$mean_trust_in_sp1
  
# Use trapz to calculate area
eq1_resilience_index <- (trapz(x = real_absorption_recovery_phases_df$timestep,
                                 y = real_absorption_recovery_phases_df$difference))
eq1_resilience_index

#####
# Cumulative performance (denoted as R) and termed resilience
# EQ3: \int_{t_0}^{t_h} S_R(t)dt / \int_{t_0}^{t_h} S_T(t)dt (Cumulative performance)
trust_real <- trapz(x = group_df$timestep, y = group_df$mean_trust_in_sp1)
trust_target <- trapz(x = target_trust_sp1_df$timestep, y = target_trust_sp1_df$mean_trust_in_sp1)
eq3_resilience_index = trust_real/trust_target
eq3_resilience_index

#####
## Experiment 2 (descriptive table) (if needed)
#####
# Use exp1 data to represent 'real' (forcassted) trust values over time
library(tidyverse)
path = "/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/attack_strategies/" 
trust_data <- list.files(path, pattern = "summary_stats_df_.*\\.csv$", full.names = TRUE)


# If necessary, add initial seed value over all documents to indicate simulation id
process_trust_file <- function(file) {
  # Extract the simulation_id (seed value) from the filename
  simulation_id <- gsub(".*-([0-9]+)-regagents-trust-data\\.csv$", "\\1", basename(file))
  # Read the CSV file
  df <- read.csv(file)
  # Add the simulation_id as the first column
  #df <- cbind(simulation_id = as.numeric(simulation_id), df)
  return(df)
}

# Apply the function to each file and store dataframes in a list
trust_dfs <- lapply(trust_data, process_trust_file)


# Combine all the data frames in the list into one large data frame
combined_trust_df <- do.call(rbind, trust_dfs)
na.omit(combined_trust_df)

# Describe data (mean, sd, min, max)
describe(combined_trust_df)

# Analyse resilience loss/cumulative performance
dat <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/attack_strategies/df_strategy_1_1.csv",
                header = TRUE)
colnames(dat)[c(4,5,6,7,8,9)] <- c("attack_stage_0", "attack_stage_1", "attack_stage_2",
                                                "attack_stage_3", "attack_stage_4", "attack_stage_5")
# Create attack_order dataframe with strategy_ids
strategy_id <-    c(0,1,2,3,4,5,6,7,8,9,10,11,12,13)
attack_stage_0 <- c(1,1,1,1,1,1,1,1,1,1,1, 1, 1, 1)
attack_stage_1 <- c(0,1,3,3,4,0,2,0,2,0,2, 0, 2, 0)
attack_stage_2 <- c(0,0,2,0,2,2,3,3,4,0,0, 2, 0, 2)
attack_stage_3 <- c(0,0,0,2,3,0,0,2,3,2,3, 3, 3, 3)
attack_stage_4 <- c(0,0,0,0,0,0,0,0,0,0,0, 0, 4, 0)
attack_stage_5 <- c(0,0,0,0,0,0,0,0,0,0,0, 0, 0, 4)
count_df <- data.frame(strategy_id, attack_stage_0, attack_stage_1,
                       attack_stage_2, attack_stage_3, attack_stage_4, attack_stage_5)
# Get attack order
attack_order <- filter(count_df, strategy_id == dat$strategy_id[[1]])
head(dat)
attack_durations <- dat %>% select(3:9) %>% slice(1)

# Rename columns
#colnames(attack_durations)[c(2,3,4,5,6,7)] <- c("attack_stage_0", "attack_stage_1", "attack_stage_2",
#                                                "attack_stage_3", "attack_stage_4", "attack_stage_5")

# Remove strategy_id column from both dataframes
attack_order <- attack_order %>% select(2:7)
attack_durations <- attack_durations %>% select(2:7) # ?
print(attack_order)
print(attack_durations)

# Filter out the stages with a duration of zero
active_stages <- names(attack_durations)[attack_durations > 0]
# Arrange the stages by the order specified in order_df
stages_in_order <- setNames(attack_order[active_stages], active_stages)
# Assuming attack_durations is a one-row dataframe
attack_durations <- unlist(attack_durations)
# Now order the stages by their order value and get the corresponding duration
ordered_stage_names <- names(sort(setNames(as.vector(t(stages_in_order)), colnames(stages_in_order))))
ordered_durations <- attack_durations[ordered_stage_names]
# Calculate the cumulative sum to find the starting time for each stage
start_times <- cumsum(ordered_durations)
# The start time for the first stage is 0, so we pretend 0 to the start_times
start_times <- c(0, head(start_times, -1))
# Name the start positions vector with the names of the stages
names(start_times) <- ordered_stage_names
# Calculate the ending time for each stage
end_times <- start_times + ordered_durations

# Define a function to translate attack stages to real names
translate_attack_stage <- function(stage_name) {
  if (stage_name == "attack_stage_0") {
    return("Reconnaissance")
  } else if (stage_name %in% c("attack_stage_1", "attack_stage_4")) {
    return("Cyberattack")
  } else if (stage_name %in% c("attack_stage_2", "attack_stage_5")) {
    return("Disinformation")
  } else if (stage_name == "attack_stage_3") {
    return("Combined attack")
  } else {
    return("Unknown Attack")
  }
}

# Apply the translation function to the stage names
label_names <- sapply(names(start_times), translate_attack_stage)

# Define the base plot
xmin_start <- start_times["attack_stage_0"] + ordered_durations["attack_stage_0"]
xmax_end <- max(end_times)  # End of the last attack stage

# Filter out the 'attack_stage_0' or any stage you want to exclude 
valid_stages <- names(start_times) != "attack_stage_0"

# Form dataframe of labels to add on graph
attack_data <- data.frame(
  x = rep(90, sum(valid_stages)), # Replicate the x position for each valid stage
  y = seq(0.95, by = -0.05, length.out = sum(valid_stages)), # Create a sequence for y positions
  label = label_names[valid_stages], # Use only the valid stages for labels
  xend = start_times[valid_stages], # Use only the valid stages for xend
  yend = seq(0.95, by = -0.05, length.out = sum(valid_stages)) # Create a sequence for yend positions, can be adjusted based on preference
)
attack_data




disruption_point <- t_D
departure_point <- filter(group_df, timestep == t_DP)
landing_point <- filter(group_df, timestep == t_LP)
ascent_point <- filter(group_df, timestep == t_AP)
recovery_point <- filter(group_df, timestep == t_RP)

# Combine the landing and ascent points into one data frame for plotting
special_points <- rbind(disruption_point, departure_point, landing_point, recovery_point)
special_points <- rbind(disruption_point, departure_point, landing_point, ascent_point, recovery_point)

# Add an identifier for the points and column for service_provider
special_points$label <- c("DE", "DP", "LP", "RP")
special_points$label <- c("DP", "LP", "AP", "RP")
special_points$service_provider <- c("Provider1","Provider1", "Provider1")
special_points$service_provider <- c("Provider1","Provider1", "Provider1", "Provider1")

# Assuming special_points has columns for nudge_x and nudge_y
special_points <- special_points %>%
  mutate(nudge_x = c(-5, 5, 7),  # Example values for horizontal adjustment
         nudge_y = c(-0.035, -0.035, -0.035)) # Example values for vertical adjustment
special_points <- special_points %>%
  mutate(nudge_x = c(-5, 5, 7, 7),  # Example values for horizontal adjustment
         nudge_y = c(-0.035, -0.035, -0.035, -0.035)) # Example values for vertical adjustment
print(special_points)

