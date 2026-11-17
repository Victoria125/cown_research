# Digital dermatitis transmission: runs the whole analysis in R and prints the tables.
#
# Run from the project folder:   Rscript R/run_analysis.R
# Optional argument: a folder where the tables are also saved as CSV (used to compare with Python).
#
# Input (data/raw): infectious data_NLAS2b.xlsx and mu_0.07_agent_exposure.csv.

if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl", repos = "https://cloud.r-project.org")
source(file.path("R", "dd_functions.R"))
options(width = 200, digits = 4)

out_dir <- commandArgs(trailingOnly = TRUE)[1]
save_table <- function(table, name) {
  if (!is.na(out_dir)) write.csv(table, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}

infection <- load_infection(file.path("data", "raw", "infectious data_NLAS2b.xlsx"))
exposure <- load_exposure(file.path("data", "raw", "mu_0.07_agent_exposure.csv"))

# ---- 1. daily and weekly tables
daily <- build_daily(infection, exposure)
weekly <- build_weekly(daily, exposure)
cat("Cows:", length(unique(daily$Agent)), "| cow-days:", nrow(daily), "| new cases (cow):", sum(daily$C),
    "| new cases (leg):", sum(daily$Ct), "\n")
cat("At-risk weeks:", sum(weekly$S == 1), "| with the 7 days of exposure:", sum(weekly$S == 1 & weekly$complete), "\n\n")
save_table(daily, "daily")
save_table(weekly, "weekly")

# ---- 2. original window (1 to 7 days before the assessment), for each case definition
# min_healthy = number of healthy assessments required before a new case (1 = original definition)
case_definition <- do.call(rbind, lapply(1:3, function(k) {
  w <- build_weekly(daily, exposure, min_healthy = k)
  do.call(rbind, lapply(c(TRUE, FALSE), function(complete) {
    weeks <- at_risk_weeks(w, complete)
    bg <- fit_with_background(weeks)
    nb <- fit_without_background(weeks)
    data.frame(min_healthy = k, weeks_used = if (complete) "7 days of exposure" else "available days",
               weeks = bg$weeks, cases = bg$cases, beta_without_background = nb$beta,
               beta = bg$beta, beta_low = bg$beta_low, beta_high = bg$beta_high, lambda0 = bg$lambda0,
               p_exposure = bg$p_exposure, dAIC_vs_constant = bg$aic - (-2 * null_loglik(weeks$C) + 2))
  }))
}))
cat("Original window (days 1-7 before the assessment), model with background\n")
print(case_definition, row.names = FALSE)
save_table(case_definition, "case_definition")

# ---- 3. delay between infection and visible lesion, estimated from the data
CONTRADICTORY_COWS <- c(2611, 2678, 4065)   # LA/RA and S/I/NC disagree in the infection file
delay <- do.call(rbind, lapply(1:3, function(k) {
  all_weeks <- weeks_by_delay(daily, exposure, k)
  do.call(rbind, lapply(c(FALSE, TRUE), function(drop) {
    w <- all_weeks
    if (drop) {
      keep <- !w$weeks$Agent %in% CONTRADICTORY_COWS
      w <- list(weeks = w$weeks[keep, ], delays = w$delays, exp = w$exp[keep, , drop = FALSE])
    }
    prof <- delay_profile(w)
    if (k == 1 && !drop) save_table(prof$profile, "delay_profile_k1")
    cbind(min_healthy = k, without_contradictory_cows = drop, prof$summary, cow_bootstrap(w))
  }))
}))
cat("\nDelay (days) between the exposure window and the assessment, estimated by maximum likelihood\n")
print(delay, row.names = FALSE)
save_table(delay, "delay")
