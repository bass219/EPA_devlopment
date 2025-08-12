# Part 0: Setup
# Ensure all necessary libraries are installed and loaded
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, nnet, xgboost, zoo, nflfastR, ggimage, gt, gtExtras)

# We are loading multiple seasons to ensure we have data for training.
# The original script failed because it only loaded 2024, leaving the train set empty.
pbp <- load_pbp(2021:2024)

# ============================================================================== #
# ================= PART 1: DRIVE EP FEATURE ENGINEERING ======================= #
# ============================================================================== #

cat("Part 1: Creating Drive EP outcome variables...\n")

pbp <- pbp %>%
  group_by(game_id, drive) %>%
  mutate(Drive = cur_group_id()) %>%
  ungroup() %>%
  mutate(
    pts_end_of_drive = case_when(
      fixed_drive_result == 'Touchdown' ~ 7,
      fixed_drive_result == 'Field goal' ~ 3,
      fixed_drive_result == 'Opp touchdown' ~ -7,
      TRUE ~ 0
    ),
    outcome_drive = as.factor(case_when(
      pts_end_of_drive == 7 ~ 0,
      pts_end_of_drive == 3 ~ 1,
      pts_end_of_drive == 0 ~ 2,
      pts_end_of_drive == -7 ~ 3
    )),
    utm = as.numeric(half_seconds_remaining <= 120),
    gtg = as.numeric(goal_to_go == 1),
    era_A = as.numeric(factor(floor(season / 4)))
  ) %>%
  filter(!is.na(down), !is.na(half_seconds_remaining), !is.na(outcome_drive))

# ============================================================================== #
# ================= PART 2: CALCULATE BASELINE EPA (epa0) ====================== #
# ============================================================================== #

cat("Part 2: Calculating baseline epa0...\n")
fit_ep0_model <- function(dataset) {
  nnet::multinom(outcome_drive ~ yardline_100 + factor(down) + log(ydstogo) + half_seconds_remaining + gtg + utm, data = dataset, trace = FALSE)
}
ep0_model <- fit_ep0_model(pbp)

ep0_preds_fun <- function(model, df) {
  probs <- predict(model, newdata = df, "probs")
  ep_preds <- probs %*% c(7, 3, 0, -7)
  return(as.vector(ep_preds))
}
pbp$ep0 <- ep0_preds_fun(ep0_model, pbp)

pbp <- pbp %>%
  group_by(game_id, Drive) %>%
  mutate(epa0 = if_else(row_number() == n(), pts_end_of_drive - ep0, lead(ep0) - ep0)) %>%
  ungroup()

# ============================================================================== #
# =============== PART 3: FOUNDATIONAL TEAM QUALITY (TQ) METRICS =============== #
# ============================================================================== #

cat("Part 3: Calculating foundational TQ metrics (qbqot, oqot, dqot)...\n")
N0 <- 100
create_running_quality_metric <- function(epa_vec, season_vec, alpha = 0.99, gamma = 0.9, N0 = 100) {
  epa_vec <- c(rep(0, N0), epa_vec)
  season_vec <- c(rep(first(season_vec), N0), season_vec)
  n <- length(epa_vec)
  quality_metric <- numeric(n)
  for (k in 2:n) {
    past_epa <- epa_vec[1:(k - 1)]
    past_seasons <- season_vec[1:(k - 1)]
    alpha_weights <- alpha^((k - 2):0)
    gamma_weights <- gamma^(last(past_seasons) - past_seasons)
    weights <- alpha_weights * gamma_weights
    quality_metric[k] <- sum(weights * past_epa, na.rm = TRUE) / sum(weights, na.rm = TRUE)
  }
  return(quality_metric[(N0 + 1):n])
}

qb_quality_df <- pbp %>%
  filter(!is.na(passer_player_name) | !is.na(rusher_player_name), !is.na(epa0)) %>%
  mutate(offensive_player_name = ifelse(!is.na(passer_player_name), passer_player_name, rusher_player_name)) %>%
  filter(!is.na(offensive_player_name)) %>%
  group_by(offensive_player_name) %>%
  filter(n() >= N0) %>%
  arrange(game_date, play_id) %>%
  mutate(qbqot = create_running_quality_metric(epa0, season, N0 = N0)) %>%
  ungroup() %>%
  select(game_id, play_id, qbqot)

team_quality_df <- pbp %>%
  filter(!is.na(posteam), !is.na(defteam), !is.na(epa0)) %>%
  group_by(posteam) %>%
  arrange(game_date, play_id) %>%
  mutate(oqot = create_running_quality_metric(epa0, season)) %>%
  ungroup() %>%
  group_by(defteam) %>%
  arrange(game_date, play_id) %>%
  mutate(dqot = create_running_quality_metric(-epa0, season)) %>%
  ungroup() %>%
  select(game_id, play_id, oqot, dqot)

pbp <- pbp %>%
  left_join(qb_quality_df, by = c("game_id", "play_id")) %>%
  left_join(team_quality_df, by = c("game_id", "play_id")) %>%
  group_by(posteam) %>%
  mutate(qbqot = na.locf(qbqot, na.rm = FALSE, fromLast = TRUE), oqot = na.locf(oqot, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup() %>%
  group_by(defteam) %>%
  mutate(dqot = na.locf(dqot, na.rm = FALSE, fromLast = TRUE)) %>%
  ungroup() %>%
  mutate(across(c(qbqot, oqot, dqot), ~replace_na(., 0)))

oq_opponent <- pbp %>% select(game_id, play_id, posteam, oqot) %>% rename(oqdt = oqot)
dq_opponent <- pbp %>% select(game_id, play_id, posteam, dqot) %>% rename(dqdt = dqot)
pbp <- pbp %>%
  left_join(oq_opponent, by = c("game_id", "play_id", "defteam" = "posteam")) %>%
  left_join(dq_opponent, by = c("game_id", "play_id", "defteam" = "posteam")) %>%
  mutate(across(c(oqdt, dqdt), ~replace_na(., 0)))


# ============================================================================== #
# ================= PART 3.5: ADVANCED FEATURE ENGINEERING ===================== #
# ============================================================================== #

cat("Part 3.5: Creating advanced sequential, player, and coaching features...\n")
pbp <- pbp %>%
  mutate(
    label = as.integer(outcome_drive),
    home = as.numeric(posteam == home_team),
    posteam_spread = if_else(!is.na(spread_line), -spread_line, 0)
  ) %>%
  group_by(Drive) %>%
  mutate(
    drive_length = n(),
    drive_weight = 1 / drive_length,
    lag_yards_gained = lag(yards_gained, default = 0),
    lag_play_type = as.factor(lag(play_type, default = "start_of_drive")),
    play_in_drive = row_number(),
    first_downs_in_drive = cumsum(if_else(first_down == 1, 1, 0, missing=0))
  ) %>%
  ungroup()

rusher_quality_df <- pbp %>%
  filter(!is.na(rusher_player_name) & !is.na(epa0)) %>%
  group_by(rusher_player_name) %>%
  filter(n() >= N0) %>%
  arrange(game_date, play_id) %>%
  mutate(rusher_quality = create_running_quality_metric(epa0, season, N0 = N0)) %>%
  ungroup() %>%
  select(game_id, play_id, rusher_quality)

receiver_quality_df <- pbp %>%
  filter(!is.na(receiver_player_name) & !is.na(epa0) & pass_attempt == 1) %>%
  group_by(receiver_player_name) %>%
  filter(n() >= N0) %>%
  arrange(game_date, play_id) %>%
  mutate(receiver_quality = create_running_quality_metric(epa0, season, N0 = N0)) %>%
  ungroup() %>%
  select(game_id, play_id, receiver_quality)

kicker_quality_df <- pbp %>%
  filter(field_goal_attempt == 1 & !is.na(kicker_player_name) & !is.na(epa0)) %>%
  group_by(kicker_player_name) %>%
  filter(n() >= 20) %>%
  arrange(game_date, play_id) %>%
  mutate(kicker_quality = create_running_quality_metric(epa0, season, N0 = 20)) %>%
  ungroup() %>%
  select(game_id, play_id, kicker_quality)

punter_quality_df <- pbp %>%
  filter(punt_attempt == 1 & !is.na(punter_player_name) & !is.na(epa0)) %>%
  group_by(punter_player_name) %>%
  filter(n() >= 30) %>%
  arrange(game_date, play_id) %>%
  mutate(punter_quality = create_running_quality_metric(epa0, season, N0 = 30)) %>%
  ungroup() %>%
  select(game_id, play_id, punter_quality)

fourth_down_agg_df <- pbp %>%
  filter(down == 4) %>%
  mutate(go_for_it = if_else(play_type %in% c("pass", "run"), 1, 0)) %>%
  group_by(posteam) %>%
  arrange(game_date, play_id) %>%
  mutate(
    cume_go_for_it = cumsum(go_for_it),
    cume_fourth_downs = row_number(),
    fourth_down_aggressiveness = lag(cume_go_for_it / cume_fourth_downs, default = 0.2)
  ) %>%
  ungroup() %>%
  select(game_id, play_id, fourth_down_aggressiveness)

pbp <- pbp %>%
  left_join(rusher_quality_df, by = c("game_id", "play_id")) %>%
  left_join(receiver_quality_df, by = c("game_id", "play_id")) %>%
  left_join(kicker_quality_df, by = c("game_id", "play_id")) %>%
  left_join(punter_quality_df, by = c("game_id", "play_id")) %>%
  left_join(fourth_down_agg_df, by = c("game_id", "play_id"))

pbp <- pbp %>%
  group_by(posteam) %>%
  fill(fourth_down_aggressiveness, .direction = "downup") %>%
  ungroup() %>%
  group_by(rusher_player_name) %>%
  fill(rusher_quality, .direction = "downup") %>%
  ungroup() %>%
  group_by(receiver_player_name) %>%
  fill(receiver_quality, .direction = "downup") %>%
  ungroup() %>%
  group_by(kicker_player_name) %>%
  fill(kicker_quality, .direction = "downup") %>%
  ungroup() %>%
  group_by(punter_player_name) %>%
  fill(punter_quality, .direction = "downup") %>%
  ungroup() %>%
  mutate(across(c(rusher_quality, receiver_quality, kicker_quality, punter_quality, fourth_down_aggressiveness), ~replace_na(., 0)))

rusher_star_threshold <- quantile(pbp$rusher_quality[pbp$rusher_quality != 0], 0.85, na.rm = TRUE)
receiver_star_threshold <- quantile(pbp$receiver_quality[pbp$receiver_quality != 0], 0.85, na.rm = TRUE)
pbp <- pbp %>%
  mutate(is_star_player_involved = if_else((rush_attempt == 1 & rusher_quality >= rusher_star_threshold) | (pass_attempt == 1 & receiver_quality >= receiver_star_threshold), 1, 0, missing = 0))

# ============================================================================== #
# ================= PART 4: MODEL BUILDING AND EVALUATION ====================== #
# ============================================================================== #

cat("Part 4: Building and evaluating all models...\n")
features_base <- c("yardline_100", "down", "ydstogo", "half_seconds_remaining", "score_differential", "era_A", "gtg", "utm", "posteam_timeouts_remaining", "defteam_timeouts_remaining")
features_tq <- c(features_base, "qbqot", "oqot", "dqot", "oqdt", "dqdt")
new_advanced_features <- c("lag_yards_gained", "play_in_drive", "first_downs_in_drive", "rusher_quality", "receiver_quality", "kicker_quality", "punter_quality", "pass_oe", "fourth_down_aggressiveness", "is_star_player_involved")
pbp$lag_play_type_numeric <- as.numeric(pbp$lag_play_type)
features_tq_ultimate <- c(features_tq, new_advanced_features, "lag_play_type_numeric")

train_set <- pbp %>% filter(season < 2024)
test_set <- pbp %>% filter(season == 2024)

clean_and_prep_data <- function(df, features) {
  df %>%
    mutate(across(all_of(c("roof", "surface")), ~as.numeric(as.factor(.)))) %>%
    select(all_of(c("outcome_drive", "pts_end_of_drive", "drive_weight", "Drive", features))) %>%
    mutate(across(all_of(features), as.numeric)) %>%
    mutate(across(everything(), ~replace_na(., 0)))
}
train_set_final <- clean_and_prep_data(train_set, features_tq_ultimate)
test_set_final <- clean_and_prep_data(test_set, features_tq_ultimate)

mlr_base <- nnet::multinom(outcome_drive ~ ., data = train_set_final %>% select(all_of(c("outcome_drive", features_base))), trace = FALSE, MaxNWts=2000)
mlr_tq <- nnet::multinom(outcome_drive ~ ., data = train_set_final %>% select(all_of(c("outcome_drive", features_tq))), trace = FALSE, MaxNWts=2000)

xgb_params_drive <- list(objective = "multi:softprob", num_class = 4, eta = 0.05, max_depth = 5, nrounds = 250, subsample = 0.8, colsample_bytree = 0.8)
train_xgb_tq <- xgb.DMatrix(data = as.matrix(train_set_final %>% select(all_of(features_tq_ultimate))), label = as.integer(train_set_final$outcome_drive)-1)
xgb_tq <- do.call(xgb.train, c(list(data=train_xgb_tq), xgb_params_drive))

train_xgb_tq_weighted <- xgb.DMatrix(data = as.matrix(train_set_final %>% select(all_of(features_tq_ultimate))), label = as.integer(train_set_final$outcome_drive)-1, weight = train_set_final$drive_weight)
xgb_tq_weighted <- do.call(xgb.train, c(list(data=train_xgb_tq_weighted), xgb_params_drive))

monotonic_constraints <- setNames(rep(0, length(features_tq_ultimate)), features_tq_ultimate)
monotonic_constraints[c("yardline_100", "ydstogo", "defteam_timeouts_remaining", "oqdt")] <- -1
monotonic_constraints[c("posteam_timeouts_remaining", "qbqot", "oqot", "dqdt", "lag_yards_gained", "first_downs_in_drive", "rusher_quality", "receiver_quality", "kicker_quality", "punter_quality")] <- 1
xgb_params_regression <- list(objective = "reg:squarederror", eta = 0.05, max_depth = 5, nrounds = 250, subsample = 0.8, colsample_bytree = 0.8, monotone_constraints = monotonic_constraints)
train_xgb_reg <- xgb.DMatrix(data = as.matrix(train_set_final %>% select(all_of(features_tq_ultimate))), label = train_set_final$pts_end_of_drive, weight = train_set_final$drive_weight)
xgb_regression_model <- do.call(xgb.train, c(list(data=train_xgb_reg), xgb_params_regression))

M <- 100000; phi <- 0.2
avg_plays_per_drive <- nrow(train_set_final) / length(unique(train_set_final$Drive))
num_drives_to_sample <- ceiling((M / avg_plays_per_drive) * 1.1)
all_drive_ids <- unique(train_set_final$Drive)
set.seed(2022); boot_drive_ids <- sample(all_drive_ids, size = num_drives_to_sample, replace = TRUE)
synthetic_X_full <- tibble(Drive = boot_drive_ids) %>% left_join(train_set_final, by = "Drive", relationship = "many-to-many")
synthetic_X <- synthetic_X_full %>% slice_sample(n = M)
synthetic_probs <- predict(mlr_tq, newdata = synthetic_X, "probs")
set.seed(2023); simulated_outcomes <- sapply(1:nrow(synthetic_probs), function(i) sample(0:3, size = 1, prob = synthetic_probs[i, ]))
synthetic_XY <- synthetic_X %>% select(-outcome_drive, -pts_end_of_drive) %>% mutate(outcome_drive = as.factor(simulated_outcomes), pts_end_of_drive = case_when(outcome_drive == 0 ~ 7, outcome_drive == 1 ~ 3, outcome_drive == 2 ~ 0, outcome_drive == 3 ~ -7))
total_real_weight <- sum(train_set_final$drive_weight); total_synthetic_weight <- total_real_weight * phi
synthetic_XY$drive_weight <- total_synthetic_weight / nrow(synthetic_XY)
catalytic_train_set <- bind_rows(train_set_final, synthetic_XY %>% select(all_of(colnames(train_set_final))))
train_xgb_catalytic <- xgb.DMatrix(data = as.matrix(catalytic_train_set %>% select(all_of(features_tq_ultimate))), label = as.integer(catalytic_train_set$outcome_drive)-1, weight = catalytic_train_set$drive_weight)
xgb_catalytic_model <- do.call(xgb.train, c(list(data=train_xgb_catalytic), xgb_params_drive))

cat("Evaluating all models on the test set...\n")
test_matrix_base <- as.matrix(test_set_final %>% select(all_of(features_base)))
test_matrix_tq <- as.matrix(test_set_final %>% select(all_of(features_tq)))
test_matrix_tq_ultimate <- as.matrix(test_set_final %>% select(all_of(features_tq_ultimate)))
point_values <- c(7, 3, 0, -7)

probs_mlr_base <- predict(mlr_base, newdata = test_set_final, "probs"); ep_mlr_base <- probs_mlr_base %*% point_values
probs_mlr_tq <- predict(mlr_tq, newdata = test_set_final, "probs"); ep_mlr_tq <- probs_mlr_tq %*% point_values
probs_xgb_tq <- matrix(predict(xgb_tq, test_matrix_tq_ultimate), ncol=4, byrow=T); ep_xgb_tq <- probs_xgb_tq %*% point_values
probs_xgb_tq_weighted <- matrix(predict(xgb_tq_weighted, test_matrix_tq_ultimate), ncol=4, byrow=T); ep_xgb_tq_weighted <- probs_xgb_tq_weighted %*% point_values
ep_xgb_regression <- predict(xgb_regression_model, test_matrix_tq_ultimate)
probs_xgb_catalytic <- matrix(predict(xgb_catalytic_model, test_matrix_tq_ultimate), ncol=4, byrow=T); ep_xgb_catalytic <- probs_xgb_catalytic %*% point_values

true_outcomes <- test_set_final$pts_end_of_drive
true_labels <- as.integer(test_set_final$outcome_drive)

calc_rmse <- function(preds, actual) { sqrt(mean((preds - actual)^2, na.rm = TRUE)) }
calc_logloss <- function(probs, labels) { safe_labels <- pmin(pmax(labels, 1), ncol(probs)); -mean(log(probs[cbind(1:nrow(probs), safe_labels)] + 1e-15)) }

results <- tibble(
  Model = c("Baseline MLR", "TQ MLR", "TQ XGBoost (Ultimate)", "TQ XGBoost (Weighted)", "XGBoost Regression (Constrained)", "XGBoost (Catalytic Prior)"),
  RMSE = c(calc_rmse(ep_mlr_base, true_outcomes), calc_rmse(ep_mlr_tq, true_outcomes), calc_rmse(ep_xgb_tq, true_outcomes), calc_rmse(ep_xgb_tq_weighted, true_outcomes), calc_rmse(ep_xgb_regression, true_outcomes), calc_rmse(ep_xgb_catalytic, true_outcomes)),
  LogLoss = c(calc_logloss(probs_mlr_base, true_labels), calc_logloss(probs_mlr_tq, true_labels), calc_logloss(probs_xgb_tq, true_labels), calc_logloss(probs_xgb_tq_weighted, true_labels), NA, calc_logloss(probs_xgb_catalytic, true_labels))
)
print("--- FINAL Model Comparison Results ---")
print(results %>% arrange(RMSE))

# ============================================================================== #
# ================= PART 5: BOOTSTRAPPING FOR UNCERTAINTY ======================== #
# ============================================================================== #
cat("\n--- Part 5: Beginning Bootstrap Uncertainty Quantification ---\n")
B <- 100
get_clustered_bootstrap_dataset <- function(dataset, group_var) {
  all_group_ids <- unique(dataset[[group_var]])
  boot_group_ids <- sample(all_group_ids, size = length(all_group_ids), replace = TRUE)
  boot_df <- purrr::map_dfr(boot_group_ids, ~dataset[dataset[[group_var]] == .x, ])
  return(boot_df)
}

if (!dir.exists("bootstrapped_models_drive")) { dir.create("bootstrapped_models_drive") }
for (b in 1:B) {
  cat("Training bootstrap model:", b, "/", B, "\n")
  set.seed(123 + b)
  train_boot <- get_clustered_bootstrap_dataset(train_set, "Drive")
  train_boot_final <- clean_and_prep_data(train_boot, features_tq_ultimate)
  
  train_xgb_boot_weighted <- xgb.DMatrix(
    data = as.matrix(train_boot_final %>% select(all_of(features_tq_ultimate))),
    label = as.integer(train_boot_final$outcome_drive) - 1,
    weight = train_boot_final$drive_weight
  )
  
  xgb_boot_model <- do.call(xgb.train, c(list(data=train_xgb_boot_weighted), xgb_params_drive))
  saveRDS(xgb_boot_model, file = paste0("bootstrapped_models_drive/tq_xgb_weighted_boot_", b, ".rds"))
}
cat("--- Finished training all bootstrap models. ---\n")

cat("\n--- Evaluating Bootstrap Coverage for DRIVE EP ---\n")
model_files <- list.files("bootstrapped_models_drive", full.names = TRUE)
boot_models <- lapply(model_files, readRDS)
all_probs <- lapply(boot_models, function(model) {
  matrix(predict(model, test_matrix_tq_ultimate), ncol = 4, byrow = TRUE)
})
outcome_map_drive <- c("0" = "Touchdown", "1" = "Field goal", "2" = "No Score", "3" = "Opp touchdown")
true_outcomes_str <- outcome_map_drive[as.character(true_labels - 1)]
covered_plays <- numeric(nrow(test_set_final))
for (i in 1:nrow(test_set_final)) {
  if (i %% 5000 == 0) cat("Processing play", i, "of", nrow(test_set_final), "\n")
  set.seed(456 + i)
  simulated_outcomes <- sapply(1:B, function(b) {
    prob_vector <- all_probs[[b]][i, ]; sample(0:3, size = 1, prob = prob_vector)
  })
  empirical_probs <- table(simulated_outcomes) / B
  sorted_outcomes <- names(empirical_probs)[order(empirical_probs, decreasing = TRUE)]
  sorted_probs <- empirical_probs[sorted_outcomes]
  cutoff_index <- which(cumsum(sorted_probs) >= 0.95)[1]
  prediction_set <- sorted_outcomes[1:cutoff_index]
  prediction_set_str <- outcome_map_drive[prediction_set]
  if (true_outcomes_str[i] %in% prediction_set_str) { covered_plays[i] <- 1 }
}
bootstrap_coverage <- mean(covered_plays, na.rm = TRUE)
se_coverage <- 2 * sqrt(bootstrap_coverage * (1 - bootstrap_coverage) / length(covered_plays))
cat("\n--- Bootstrap Evaluation Results (DRIVE EP) ---\n")
cat("Model: TQ XGBoost (Weighted) with Cluster Bootstrap on Drives\n")
cat("Desired Coverage Level: 95%\n")
cat("Actual Observed Coverage:", round(bootstrap_coverage, 4), " (±", round(se_coverage, 4), ")\n")



# ============================================================================== #
# == PART 5: UNCERTAINTY QUANTIFICATION WITH CLUSTER BOOTSTRAPPING ON DRIVES ===== #
# ============================================================================== #

# --- Step 1: Setup and Bootstrap Data Generation ---
B <- 100 

# Create helper function for cluster bootstrapping on DRIVES
get_clustered_bootstrap_dataset <- function(dataset, group_var) {
  all_group_ids <- unique(dataset[[group_var]])
  boot_group_ids <- sample(all_group_ids, size = length(all_group_ids), replace = TRUE)
  boot_df <- purrr::map_dfr(boot_group_ids, ~dataset[dataset[[group_var]] == .x, ])
  return(boot_df)
}

# --- Step 2: Train B Models on Bootstrapped Datasets ---
cat("\n--- Starting Bootstrap Training Loop for B =", B, "DRIVE EP models ---\n")
if (!dir.exists("bootstrapped_models_drive")) { dir.create("bootstrapped_models_drive") }

for (b in 1:B) {
  cat("Training bootstrap model:", b, "/", B, "\n")
  
  set.seed(123 + b)
  # Bootstrap by resampling DRIVES from the full training set
  train_boot <- get_clustered_bootstrap_dataset(train_set, "Drive")
  
  # Clean and prepare this specific bootstrap sample
  train_boot_tq <- clean_and_prep_drive_data(train_boot, features_tq)
  
  train_xgb_boot_weighted <- xgb.DMatrix(
    data = as.matrix(train_boot_tq %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight)),
    label = as.integer(train_boot_tq$outcome_drive) - 1,
    weight = train_boot_tq$drive_weight
  )
  
  xgb_boot_model <- xgb.train(params = xgb_params_drive, data = train_xgb_boot_weighted, nrounds = 200, verbose = 0)
  
  saveRDS(xgb_boot_model, file = paste0("bootstrapped_models_drive/tq_xgb_weighted_boot_", b, ".rds"))
}
cat("--- Finished training all bootstrap models. ---\n")


# --- Step 3: Evaluate Bootstrap Coverage on the Test Set ---
cat("\n--- Evaluating Bootstrap Coverage for DRIVE EP ---\n")

model_files <- list.files("bootstrapped_models_drive", full.names = TRUE)
boot_models <- lapply(model_files, readRDS)

# Get probability predictions from ALL B models
all_probs <- lapply(boot_models, function(model) {
  matrix(predict(model, test_matrix_tq_for_weighted), ncol = 4, byrow = TRUE)
})

# Get true outcomes for the test set
true_labels <- as.integer(test_set_tq_drive$outcome_drive)
# Use the DRIVE outcome map
outcome_map_drive <- c("0" = "Touchdown", "1" = "Field goal", "2" = "No Score", "3" = "Opp touchdown")
true_outcomes_str <- outcome_map_drive[as.character(true_labels - 1)]

covered_plays <- numeric(nrow(test_set_tq_drive))

for (i in 1:nrow(test_set_tq_drive)) {
  if (i %% 1000 == 0) cat("Processing play", i, "of", nrow(test_set_tq_drive), "\n")
  
  set.seed(456 + i)
  simulated_outcomes <- sapply(1:B, function(b) {
    prob_vector <- all_probs[[b]][i, ]
    sample(0:3, size = 1, prob = prob_vector) # Sample from 4 outcomes
  })
  
  empirical_probs <- table(simulated_outcomes) / B
  sorted_outcomes <- names(empirical_probs)[order(empirical_probs, decreasing = TRUE)]
  sorted_probs <- empirical_probs[sorted_outcomes]
  
  cutoff_index <- which(cumsum(sorted_probs) >= 0.95)[1]
  prediction_set <- sorted_outcomes[1:cutoff_index]
  prediction_set_str <- outcome_map_drive[prediction_set]
  
  if (true_outcomes_str[i] %in% prediction_set_str) {
    covered_plays[i] <- 1
  }
}

# --- Final Step: Calculate and Display Bootstrap Coverage ---
bootstrap_coverage <- mean(covered_plays, na.rm = TRUE)
se_coverage <- 2 * sqrt(bootstrap_coverage * (1 - bootstrap_coverage) / length(covered_plays))

cat("\n--- Bootstrap Evaluation Results (DRIVE EP) ---\n")
cat("Model: TQ XGBoost (Weighted) with Cluster Bootstrap on Drives\n")
cat("Desired Coverage Level: 95%\n")
cat("Actual Observed Coverage:", round(bootstrap_coverage, 4), " (±", round(se_coverage, 4), ")\n")


# ============================================================================== #
# ======== PART 4.6: XGBOOST REGRESSION WITH MONOTONIC CONSTRAINTS =============== #
# ============================================================================== #

# This section implements the methodology from Section 4.1 of the paper to
# directly model EP via regression and enforce logical constraints to reduce overfitting.

print("Training Model 6: XGBoost Regression (Weighted & Constrained)")

# --- Step 1: Define the Monotonic Constraints ---
# We create a named vector. 1 for increasing, -1 for decreasing, 0 for no constraint.
# These relationships are based on football logic as described in the paper.
monotonic_constraints <- c(
  yardline_100 = -1,               # EP decreases as you get further from the end zone
  down = 0,                        # Not strictly monotonic (e.g., 4th down can be > 3rd)
  ydstogo = -1,                    # EP decreases as yards to go increases
  half_seconds_remaining = 0,      # Not strictly monotonic (value can be low at start and end of half)
  score_differential = 0,          # Not monotonic (e.g., being up 3 vs up 28)
  posteam_timeouts_remaining = 1,  # EP increases with more timeouts
  defteam_timeouts_remaining = -1, # EP decreases as the defense has more timeouts
  era_A = 0,
  gtg = 0,
  utm = 0,
  posteam_spread = -1,             # EP decreases for bigger underdogs (more negative spread)
  total_line = 0,
  xpass = 0,
  shotgun = 0,
  no_huddle = 0,
  qb_dropback = 0,
  roof = 0,
  surface = 0,
  qbqot = 1,                       # EP increases with better QB quality
  oqot = 1,                        # EP increases with better Offense quality
  dqot = -1,                       # EP decreases with better Defense quality
  oqdt = -1,                       # EP decreases facing a better Offense
  dqdt = 1                         # EP increases facing a worse Defense
)

# Ensure the constraints vector matches the feature set of your best model (TQ Ultimate)
# We need to get the feature names from the prepared data to match the order
feature_names_for_regression <- colnames(train_set_tq_drive %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight))
constraints_vector <- monotonic_constraints[feature_names_for_regression]


# --- Step 2: Create the Training and Testing DMatrix for Regression ---
# The LABEL is now the continuous point value, not the categorical outcome.
train_xgb_reg <- xgb.DMatrix(
  data = as.matrix(train_set_tq_drive %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight)),
  label = train_set_tq_drive$pts_end_of_drive, # Use the numeric points as the label
  weight = train_set_tq_drive$drive_weight
)

test_matrix_reg <- as.matrix(test_set_tq_drive %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight))


# --- Step 3: Train the XGBoost Regression Model ---
# We change the objective to 'reg:squarederror' and add the constraints.
xgb_params_regression <- list(
  objective = "reg:squarederror", # Regression objective
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  monotone_constraints = constraints_vector # Apply the constraints here
)

xgb_regression_model <- xgb.train(
  params = xgb_params_regression,
  data = train_xgb_reg,
  nrounds = 200,
  verbose = 0
)


# --- Step 4: Evaluate the Regression Model and Add to Results ---
# The prediction is now a single numeric value (the EP), not a matrix of probabilities.
ep_xgb_regression <- predict(xgb_regression_model, test_matrix_reg)

# Add the regression model's results to your table
# Note: LogLoss is not applicable to a regression model, so we'll put NA.
results <- results %>%
  add_row(
    Model = "XGBoost Regression (Constrained)",
    RMSE = calc_rmse(ep_xgb_regression, test_set_tq_drive$pts_end_of_drive),
    LogLoss = NA
  )

print("--- FINAL Model Comparison Results (with Regression Model) ---")
print(results %>% arrange(RMSE))

# ============================================================================== #
# ================= PART 7: SMOOTHING XGBOOST WITH A CATALYTIC PRIOR ============== #
# ============================================================================== #

print("Beginning Catalytic Prior modeling process...")
M <- 100000 
phi <- 0.2   

# --- Step 2 (Efficient Version): Generate Synthetic Game States (X) ---

cat("Generating", M, "synthetic game states by resampling drives...\n")

# This will now work correctly because train_set_tq_drive contains the 'Drive' column
avg_plays_per_drive <- nrow(train_set_tq_drive) / length(unique(train_set_tq_drive$Drive))
num_drives_to_sample <- ceiling((M / avg_plays_per_drive) * 1.1)
cat("Estimated drives to sample:", num_drives_to_sample, "\n")
all_drive_ids <- unique(train_set_tq_drive$Drive)
set.seed(2022)
boot_drive_ids <- sample(all_drive_ids, size = num_drives_to_sample, replace = TRUE)
synthetic_X_full <- tibble(Drive = boot_drive_ids) %>%
  left_join(train_set_tq_drive, by = "Drive")
synthetic_X <- synthetic_X_full %>%
  slice_sample(n = M)
cat("Successfully generated", nrow(synthetic_X), "synthetic plays.\n")

# --- Step 3: Impute Synthetic Outcomes (Y) using the MLR Prior ---

cat("Imputing synthetic outcomes using the TQ MLR model as the prior...\n")

# Use the trained `mlr_tq` model to get outcome probabilities for the synthetic game states
synthetic_probs <- predict(mlr_tq, newdata = synthetic_X, "probs")

# Simulate one outcome for each synthetic play based on the MLR's predicted probabilities
set.seed(2023)
simulated_outcomes <- sapply(1:nrow(synthetic_probs), function(i) {
  sample(0:3, size = 1, prob = synthetic_probs[i, ])
})

# Create the final synthetic dataset (synthetic_XY)
synthetic_XY <- synthetic_X %>%
  select(-outcome_drive, -pts_end_of_drive) %>% # Remove the original (real) outcomes
  mutate(
    # Add the new, simulated outcomes
    outcome_drive = as.factor(simulated_outcomes),
    # Add the corresponding point values for the new outcomes
    pts_end_of_drive = case_when(
      outcome_drive == 0 ~ 7,
      outcome_drive == 1 ~ 3,
      outcome_drive == 2 ~ 0,
      outcome_drive == 3 ~ -7
    )
  )


# --- Step 4: Combine Datasets and Train the Final Catalytic Model ---

cat("Combining real and synthetic data to train the final catalytic model...\n")

# Calculate the weights for the synthetic data based on phi
total_real_weight <- sum(train_set_tq_drive$drive_weight)
total_synthetic_weight <- total_real_weight * phi

# Assign the calculated weight to each row of the synthetic data
synthetic_XY$drive_weight <- total_synthetic_weight / nrow(synthetic_XY)

# Combine the real training data with the new synthetic data
# We only need the columns required for training
cols_to_keep <- colnames(train_set_tq_drive)
catalytic_train_set <- bind_rows(
  train_set_tq_drive,
  synthetic_XY %>% select(all_of(cols_to_keep))
)

# Create the final weighted DMatrix for the catalytic model
train_xgb_catalytic <- xgb.DMatrix(
  data = as.matrix(catalytic_train_set %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight)),
  label = as.integer(catalytic_train_set$outcome_drive) - 1,
  weight = catalytic_train_set$drive_weight
)

# Train the final model
# Using the same parameters as our other Drive EP XGBoost models
xgb_catalytic_model <- xgb.train(
  params = xgb_params_drive,
  data = train_xgb_catalytic,
  nrounds = 200,
  verbose = 0
)


# --- Step 5: Evaluate the Catalytic Model and Add to Results ---
print("Evaluating Catalytic Model...")

# The prediction matrix is the same as the others
test_matrix_for_catalytic <- as.matrix(test_set_tq_drive %>% select(-outcome_drive, -pts_end_of_drive, -drive_weight))

# Get predictions
probs_xgb_catalytic <- matrix(predict(xgb_catalytic_model, test_matrix_for_catalytic), ncol = 4, byrow = TRUE)
ep_xgb_catalytic <- probs_xgb_catalytic %*% c(7, 3, 0, -7)

# Add the catalytic model's results to your table
results <- results %>%
  add_row(
    Model = "XGBoost (Catalytic Prior)",
    RMSE = calc_rmse(ep_xgb_catalytic, test_set_tq_drive$pts_end_of_drive),
    LogLoss = calc_logloss(probs_xgb_catalytic, as.integer(test_set_tq_drive$outcome_drive))
  )

print("--- FINAL Model Comparison Results (with Catalytic Model) ---")
print(results %>% arrange(RMSE))