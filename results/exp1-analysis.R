# Experiment 1

# Load libraries
library(psych)
library(dplyr)
library(tidyverse)

# Use exp1 data to represent 'real' (forcassted) trust values over time
path = "exp1-results"
trust_data <- list.files(path, pattern = "*-regagents-trust-data\\.csv$", full.names = TRUE)

# If necessary, add initial seed value over all documents to indicate simulation id
process_trust_file <- function(file) {
  # Extract the simulation_id (seed value) from the filename
  simulation_id <- gsub(".*-([0-9]+)-regagents-trust-data\\.csv$", "\\1", basename(file))
  # Read the CSV file
  df <- read.csv(file)
  # Add the simulation_id as the first column
  df <- cbind(simulation_id = as.numeric(simulation_id), df)
  return(df)
}

# Apply the function to each file and store dataframes in a list
trust_dfs <- lapply(trust_data, process_trust_file)

# Combine all the data frames in the list into one large data frame
combined_trust_df <- do.call(rbind, trust_dfs)

# Describe data (mean, sd, min, max)
describe(combined_trust_df)

# List of columns for which to calculate mean and sd
trust_columns <- c("group0_trust_in_sp0", "group0_trust_in_sp1", "group0_trust_in_sp2", 
                   "group1_trust_in_sp0", "group1_trust_in_sp1", "group1_trust_in_sp2", 
                   "group2_trust_in_sp0", "group2_trust_in_sp1", "group2_trust_in_sp2", 
                   "group3_trust_in_sp0", "group3_trust_in_sp1", "group3_trust_in_sp2")

# Calculate mean and sd for each timestep
summary_df <- combined_trust_df %>%
  group_by(timestep) %>%
  summarise(across(all_of(trust_columns), list(mean = mean, sd = sd), .names = "{.col}.{.fn}"))

# Print result
print(summary_df)

# Write summary_stats to csv
write.csv(summary_df, "exp1-results/summary_regagents_trust.csv", row.names=FALSE)

#####
# Visualise data
#####
df_long <- summary_df %>%
  pivot_longer(
    cols = -timestep,
    names_to = c("group", "service_provider", ".value"),
    names_pattern = "(group[0-3])_trust_in_(sp\\d).(mean|sd)"
  )

# Recode the service_provider values
df_long$service_provider[df_long$service_provider=="sp0"] <- "Service provider 1"
df_long$service_provider[df_long$service_provider=="sp1"] <- "Service provider 2"
df_long$service_provider[df_long$service_provider=="sp2"] <- "Service provider 3"

# Plot data
p <- function(group_data) {
  # Custom colors
  custom_colors <- c("Service provider 1" = "#1F77B4", "Service provider 2" = "#FF7F0E", 
                     "Service provider 3" = "#2CA02C")
  
  ggplot(group_data, aes(x = timestep, y = mean, group = service_provider, 
                         color = service_provider)) +
    geom_line() +
    geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = service_provider), 
                alpha = 0.2, linetype = 0) +
    theme(legend.title = element_blank()) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_continuous(limits = c(0, 500)) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    labs(title = "Trust in service providers",
         x = "Timestep",
         y = "Trust")
}

g0 <- p(df_long %>% filter(group == "group0"))
g0

g1 <- p(df_long %>% filter(group == "group1"))
g1

g2 <- p(df_long %>% filter(group == "group2"))
g2

g3 <- p(df_long %>% filter(group == "group3"))
g3
