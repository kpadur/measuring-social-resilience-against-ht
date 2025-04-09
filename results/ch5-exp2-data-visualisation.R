# Analyse resilience
library(tidyverse)
library(mcp)
library(pracma)
library(psych)

# Read attack strategy from file
dat <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2-results/attack_strategies/df_strategy_1.csv", 
                header = TRUE)
# Get attackers data
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
colnames(attack_durations)[c(2,3,4,5,6,7)] <- c("attack_stage_0", "attack_stage_1", "attack_stage_2",
                                                "attack_stage_3", "attack_stage_4", "attack_stage_5")

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

# Get regular agents data
# Use exp1 data to represent 'real' (forcasted) trust values over time
targeted_trust <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp1-results/summary_regagents_trust.csv",
                           header = TRUE)

# Find max and min trust values specifically for Provider 2 for every group (not including time step 0)
summary_stats <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2-results/attack_strategies/summary_stats_df_strategy_1.csv",
                          header = TRUE)
durations <- read.csv("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/attack_strategies/df_time_durations.csv", 
                 header=TRUE, sep = ",")

# Analyse specific group's resilience
group_sp1_df <- summary_stats %>% select(timestep, group3_trust_in_sp1_mean)
# Rename columns
colnames(group_sp1_df)[c(1,2)] <- c("timestep", "mean_trust_in_sp1")

t_DP <- 100
t_LP <- 121
t_AP <- 200
t_RP <- 237

disruption_point <- filter(group_sp1_df, timestep == t_DP)
landing_point <- filter(group_sp1_df, timestep == t_LP)
ascent_point <- filter(group_sp1_df, timestep == t_AP)
recovery_point <- filter(group_sp1_df, timestep == t_RP)

# Combine the landing and ascent points into one data frame for plotting
special_points <- rbind(disruption_point, landing_point, ascent_point, recovery_point)

# Add an identifier for the points and column for service_provider
special_points$label <- c("DP", "LP", "AP", "RP")
special_points$service_provider <- c("Provider1","Provider1", "Provider1", "Provider1")

# Assuming special_points has columns for nudge_x and nudge_y
special_points <- special_points %>%
  mutate(nudge_x = c(-5, 5, 7, 7),  # Example values for horizontal adjustment
         nudge_y = c(-0.035, -0.035, -0.035, -0.035)) # Example values for vertical adjustment
print(special_points)

# The following needs to be separately done for each group - CHANGE:group id
# Filter and reshape data for group0
summary_stats <- summary_stats %>% 
  rename(
    group3_trust_in_sp0.mean = group3_trust_in_sp0_mean,
    group3_trust_in_sp1.mean = group3_trust_in_sp1_mean,
    group3_trust_in_sp2.mean = group3_trust_in_sp2_mean,
    group3_trust_in_sp0.sd = group3_trust_in_sp0_sd,
    group3_trust_in_sp1.sd = group3_trust_in_sp1_sd,
    group3_trust_in_sp2.sd = group3_trust_in_sp2_sd
  )

group_data <- summary_stats %>%
  select(timestep, group3_trust_in_sp0.mean, group3_trust_in_sp0.sd, 
                  group3_trust_in_sp1.mean, group3_trust_in_sp1.sd, 
                  group3_trust_in_sp2.mean, group3_trust_in_sp2.sd) %>%
  gather(key = "variable", value = "value", -timestep) %>%
  separate(variable, into = c("service_provider", "statistic"), sep = "\\.") %>%
  spread(statistic, value)

# Rename service providers
group_data$service_provider <- factor(group_data$service_provider,
                  levels = c("group3_trust_in_sp0", 
                             "group3_trust_in_sp1", 
                             "group3_trust_in_sp2"),
                  labels = c("Provider 1", "Provider 2", "Provider 3"))

# Visualise data
p1 <- function(group_data, attack_data, special_points) {
  
  # Custom colors
  custom_colors <- c("Provider 1" = "#1F77B4", "Provider 2" = "#FF7F0E", "Provider 3" = "#2CA02C")
  
  # Plot
  ggplot(group_data, aes(x = timestep, y = mean, group = service_provider, color = service_provider)) +
    geom_rect(aes(xmin = xmin_start, xmax = xmax_end, ymin = -Inf, ymax = Inf), 
              fill = "#FFCCCC", alpha = 0.2, color = NA) +  # Add the transparent red area
    geom_line() +
    geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd, fill = service_provider), alpha = 0.2, linetype = 0) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    scale_y_continuous(limits = c(0, 1)) +
    scale_x_continuous(limits = c(0, 500)) +
    
    theme(legend.title = element_blank()) +
    
    # Attack data
    geom_vline(data = attack_data, aes(xintercept = xend), linetype = "dashed", color = "red") +
    geom_segment(data = attack_data, aes(x = x, xend = xend, y = y, yend = yend, group = label),
                 arrow = arrow(type = "open", angle = 15, length = unit(0.1, "inches")),
                 color = "black") +
    geom_text(data = attack_data, aes(x = x, y = y, label = label, group = label), vjust = 0.5, hjust = 1, color = "black", size = 3) +
    geom_point(data = special_points, aes(x = timestep, y = mean_trust_in_sp1, group = service_provider), color = "black", size = 2) +
    geom_text(data = special_points, aes(x = timestep, y = mean_trust_in_sp1, label = label),
              nudge_x = special_points$nudge_x, nudge_y = special_points$nudge_y, color = "black", size = 3) +
    labs(x = "Time step", y = "Trust") + 
    expand_limits(x = c(0, 500), y = c(0, 1)) +
    theme(
      legend.position = c(0.80, 0.31),
      plot.title = element_text(size = rel(1)),
      axis.title = element_text(size = rel(1)),
      axis.text = element_text(size = rel(1)),
      legend.text = element_text(size = rel(0.80)),
      legend.background = element_blank(),
      legend.key = element_rect(fill = NA, color = NA),  # Ensure the legend key has no fill or border
      plot.margin = margin(1, 1, 1, 1, "cm"),
      axis.title.x = element_text(vjust = -1),
      axis.title.y = element_text(vjust = 1),
      panel.background = element_rect(fill = "white", color = NA),  # Set panel background to white
      panel.grid.major = element_line(color = "grey", size = 0.5), 
      panel.grid.minor = element_line(color = "lightgrey", size = 0.5),
      panel.border = element_rect(color = "black", fill = NA, size = 0.51)  # Add black border around the plot area
    )
}

# Example usage:
# p1(group_data, attack_data, special_points)


# Example usage:
# p1(group_data, attack_data, special_points)



# Example usage:
# p1(group_data, attack_data, special_points)


g0 <- p1(group_data, attack_data, special_points)
g0
#ggsave("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/plots/....png", 
#       plot = g0, width = 12, height = 6, dpi = 300, bg = "white")

g1 <- p1(group_data, attack_data, special_points)
g1
#ggsave("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/plots/....png", 
#       plot = g1, width = 12, height = 6, dpi = 300, bg = "white")

g2 <- p1(group_data, attack_data, special_points)
g2
#ggsave("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/plots/....png",
#       plot = g2, width = 12, height = 6, dpi = 300, bg = "white")


g3 <- p1(group_data, attack_data, special_points)
g3
#ggsave("/Users/kartpadur/Documents/GitHub/pytorch_project/ch5-resilience/results/exp2_data/plots/....png", 
#       plot = g3, width = 12, height = 6, dpi = 300, bg = "white")

