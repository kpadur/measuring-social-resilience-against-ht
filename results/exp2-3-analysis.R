# Experiment 2 and 3
# Import libraries
library(tidyverse)
library(mcp)
library(rjags)
library(pracma)

# Resilience analysis
# Piece-wise linear regression analysis
# Use exp1 data to represent 'real' (forcasted) trust values over time
targeted_trust <- read.csv("exp1-results/summary_regagents_trust.csv", header = TRUE)
# Use exp2 data to represent 'observed' trust values over time
summary_stats <- read.csv("exp2-results/attack-strategies/summary_stats_df_strategy_1.csv",
                          header = TRUE)
# Use exp3 data to represent 'observed' trust values over time
summary_stats <- read.csv("exp3-results/attack-strategies/summary_stats_df_strategy_1.csv",
                          header = TRUE)

# Determine change points in data
warm_up_period <- 0
end <- 500
group_df <- summary_stats %>% select(timestep, group0_trust_in_sp1_mean)
colnames(group_df)[c(1,2)] <- c("timestep", "mean_trust_in_sp1")
target_trust_sp1_df <- targeted_trust %>% select(timestep, group0_trust_in_sp1.mean)
colnames(target_trust_sp1_df)[c(1,2)] <- c("timestep", "mean_trust_in_sp1")

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
           iter = 10000)
# View summary
summary(fit)
plot(fit) + plot_pars(fit, pars = c("cp_1", "cp_2", "cp_3", "cp_4"), type = "dens_overlay")

# Determine disruption point, landing point, and recovery point
summary_fit <- summary(fit)

t_DP <- round(summary_fit$mean[summary_fit$name == "cp_1"])
t_LP <- round(summary_fit$mean[summary_fit$name == "cp_2"])
t_AP <- round(summary_fit$mean[summary_fit$name == "cp_3"])
t_RP <- round(summary_fit$mean[summary_fit$name == "cp_4"])
# Record data in csv
print(t_DP)
print(t_LP)
print(t_AP)
print(t_RP)

#####
# Resilience loss calculated manually
#####
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
