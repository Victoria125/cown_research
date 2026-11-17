# Digital dermatitis transmission: functions (R version of the Python code in src/).
#
# Pipeline: raw files -> daily table -> weekly table -> regression.
# Only base R and the readxl package are used.
#
# Model (Chang & de Jong 2023), per cow per assessment week:
#   without background:  p = 1 - exp(-beta * exp)                 (binomial GLM, cloglog link, offset log(exp))
#   with background:     p = 1 - exp(-(lambda0 * 7 + beta * exp))  (lambda0 = risk per day that does not come from the floor)

ASSESSMENT_WDAY <- 4   # Thursday in as.POSIXlt()$wday (0 = Sunday)
WINDOW_DAYS <- 7
CI_DROP <- qchisq(0.95, 1) / 2   # drop in log-likelihood for a 95% profile interval

# ---------------------------------------------------------------- 1. raw files

# Leg status per cow per day (LA/RA). The NC, S and I columns of the file are not used:
# the status is recalculated from LA and RA.
load_infection <- function(path) {
  raw <- as.data.frame(readxl::read_excel(path))
  data.frame(Agent = as.integer(raw$ID), barn = as.integer(raw$Barn),
             date = as.Date(raw$date), LA = as.integer(raw$LA), RA = as.integer(raw$RA))
}

# Daily exposure per cow from PeDViS-BDD for one value of mu.
load_exposure <- function(path) {
  raw <- read.csv(path, check.names = FALSE)
  data.frame(Agent = as.integer(raw$Agent), date = as.Date(raw$Date, format = "%d-%m-%Y"),
             Exp = raw[["Contamination Load Droplet"]])
}

# ---------------------------------------------------------------- 2. daily table

# Value of the previous row of the same cow (NA on the first row of each cow).
# The table must be sorted by Agent and date.
lag_by_cow <- function(x, agent, n = 1) {
  out <- c(rep(NA, n), x[seq_len(length(x) - n)])
  out[c(rep(NA, n), agent[seq_len(length(agent) - n)]) != agent | seq_along(x) <= n] <- NA
  out
}

# Removes repeated rows and solves barn-move days. On a move day the cow appears in two
# barns; the destination barn (the barn of the next day without a move) is kept.
one_row_per_day <- function(infection) {
  df <- unique(infection)
  df <- df[order(df$Agent, df$date), ]
  key <- paste(df$Agent, df$date)
  legs <- unique(df[c("Agent", "date", "LA", "RA")])
  if (anyDuplicated(legs[c("Agent", "date")])) stop("Different LA/RA for the same cow on the same day")

  n_barns <- tapply(df$barn, key, function(b) length(unique(b)))
  days <- df[!duplicated(key), ]
  days$barn_move <- as.vector(n_barns[paste(days$Agent, days$date)]) > 1

  # destination = barn of the next day without a move (fill backwards within the cow)
  next_barn <- ifelse(days$barn_move, NA, days$barn)
  for (i in rev(seq_len(nrow(days) - 1))) {
    if (is.na(next_barn[i]) && days$Agent[i] == days$Agent[i + 1]) next_barn[i] <- next_barn[i + 1]
  }
  moved <- which(days$barn_move)
  ok <- !is.na(next_barn[moved]) &
    paste(days$Agent[moved], days$date[moved], next_barn[moved]) %in% paste(df$Agent, df$date, df$barn)
  if (!all(ok)) stop("Barn move without a clear destination")
  days$barn[moved] <- next_barn[moved]
  rownames(days) <- NULL
  days
}

# INFECTED, C, I, S, N and Ct, comparing each day with the previous observed day of the same cow
# (rules of Gustavo's e-mail of 16 Oct):
# - first day of a cow: no previous day, so C = 0 and Ct = 0; if already infected it counts as I;
# - second leg of an already infected cow, or a leg switch on the same day: Ct = 1 but C = 0.
add_status <- function(df) {
  df <- df[order(df$Agent, df$date), ]
  df$INFECTED <- as.integer(df$LA == 1 | df$RA == 1)
  prev_date <- lag_by_cow(df$date, df$Agent)
  prev_la <- lag_by_cow(df$LA, df$Agent)
  prev_ra <- lag_by_cow(df$RA, df$Agent)
  prev_inf <- lag_by_cow(df$INFECTED, df$Agent)

  df$first_day <- is.na(prev_inf)
  df$days_since_prev <- as.integer(df$date - as.Date(prev_date, origin = "1970-01-01"))
  df$at_risk <- !is.na(prev_inf) & prev_inf == 0            # healthy on the previous day
  df$legs_at_risk <- as.integer(!is.na(prev_la) & prev_la == 0) + as.integer(!is.na(prev_ra) & prev_ra == 0)
  df$C <- as.integer(df$at_risk & df$INFECTED == 1)
  df$I <- as.integer(df$INFECTED == 1 & df$C == 0)
  df$S <- 1L - df$INFECTED
  df$N <- 1L
  df$Ct <- as.integer(df$LA == 1 & !is.na(prev_la) & prev_la == 0) +
    as.integer(df$RA == 1 & !is.na(prev_ra) & prev_ra == 0)
  rownames(df) <- NULL
  df
}

# One row per cow per day, in the format of daily000000007.xlsx plus helper columns.
# Cow-days without exposure keep Exp = NA; cows that only have exposure are left out.
build_daily <- function(infection, exposure) {
  daily <- add_status(one_row_per_day(infection))
  daily$Exp <- exposure$Exp[match(paste(daily$Agent, daily$date), paste(exposure$Agent, exposure$date))]
  daily[c("Agent", "barn", "date", "Exp", "LA", "RA", "INFECTED", "C", "I", "S", "N", "Ct",
          "barn_move", "first_day", "days_since_prev", "at_risk", "legs_at_risk")]
}

# ---------------------------------------------------------------- 3. weekly table

# Sum of Exp over the 7 days that end `delay` days before each date (the day itself is left out).
# delay = 0: days 1 to 7 before the assessment; delay = 14: days 15 to 21 before.
exposure_window <- function(agent, date, exposure, days = WINDOW_DAYS, delay = 0) {
  exp_key <- paste(exposure$Agent, exposure$date)
  total <- numeric(length(agent))
  n <- integer(length(agent))
  for (lag in (delay + 1):(delay + days)) {
    v <- exposure$Exp[match(paste(agent, date - lag), exp_key)]
    has <- !is.na(v)
    total[has] <- total[has] + v[has]
    n <- n + has
  }
  total[n == 0] <- NA
  data.frame(exp = total, exp_days = n)
}

# TRUE if the cow was healthy at the `min_healthy` previous checks. The checks are the Thursdays
# and the first day of the cow. With min_healthy = 1 this is "healthy on the day before".
healthy_before <- function(daily, min_healthy) {
  is_check <- as.POSIXlt(daily$date)$wday == ASSESSMENT_WDAY | daily$first_day
  checks <- daily[is_check, ]
  healthy <- rep(TRUE, nrow(checks))
  for (j in seq_len(min_healthy)) {
    prev <- lag_by_cow(checks$INFECTED, checks$Agent, j)
    healthy <- healthy & !is.na(prev) & prev == 0
  }
  out <- rep(FALSE, nrow(daily))
  out[is_check] <- healthy
  out
}

# One row per cow per assessment (Thursday): the unit of the regression.
#   S = at risk (healthy at the min_healthy previous checks), C = new case at the assessment,
#   exp = exposure summed over the 7-day window, exp_days = days with exposure in the window.
# `exposure` should be the whole exposure file, so days missing from the infection file still count.
build_weekly <- function(daily, exposure, min_healthy = 1, delay = 0) {
  daily <- daily[order(daily$Agent, daily$date), ]
  prev_la <- lag_by_cow(daily$LA, daily$Agent)
  prev_ra <- lag_by_cow(daily$RA, daily$Agent)
  thursday <- as.POSIXlt(daily$date)$wday == ASSESSMENT_WDAY
  changed <- !daily$first_day & (daily$LA != prev_la | daily$RA != prev_ra)
  if (any(changed & !thursday)) stop("Leg status changed outside the assessment day")

  daily$S <- as.integer(healthy_before(daily, min_healthy))
  weekly <- daily[thursday & !daily$first_day, c("Agent", "barn", "date", "S", "C", "legs_at_risk", "Ct")]
  weekly <- cbind(weekly, exposure_window(weekly$Agent, weekly$date, exposure, delay = delay))
  weekly$complete <- weekly$exp_days == WINDOW_DAYS
  rownames(weekly) <- NULL
  weekly
}

# Weeks in which the cow was at risk and has exposure. complete = TRUE requires all 7 days.
at_risk_weeks <- function(weekly, complete = TRUE) {
  keep <- weekly$S == 1 & weekly$exp_days > 0
  if (complete) keep <- keep & weekly$complete
  weekly[keep, ]
}

# ---------------------------------------------------------------- 4. regression

loglik <- function(lambda0, beta, case, exp) {
  h <- lambda0 * WINDOW_DAYS + beta * exp
  sum(log(-expm1(-h[case == 1]))) - sum(h[case == 0])
}

null_loglik <- function(case) {
  rate <- mean(case)
  sum(case) * log(rate) + sum(1 - case) * log(1 - rate)
}

# Without background: the cloglog GLM with offset log(exp) of the paper, on the weeks with exp > 0.
fit_without_background <- function(weeks) {
  used <- weeks[weeks$exp > 0, ]
  fit <- glm(C ~ 1 + offset(log(exp)), family = binomial(link = "cloglog"), data = used)
  ll <- as.numeric(logLik(fit))
  log_beta <- unname(coef(fit)[1])
  above <- function(lb) loglik(0, exp(lb), used$C, used$exp) - (ll - CI_DROP)
  data.frame(model = "without background", weeks = nrow(used), cases = sum(used$C),
             beta = exp(log_beta),
             beta_low = exp(uniroot(above, c(log_beta - 20, log_beta), tol = 1e-10)$root),
             beta_high = exp(uniroot(above, c(log_beta, log_beta + 20), tol = 1e-10)$root),
             lambda0 = NA, loglik = ll, aic = -2 * ll + 2, p_exposure = NA)
}

# With background: maximum likelihood of lambda0 and beta (both >= 0). The log-likelihood is
# concave in (lambda0, beta), so there is a single maximum. Returns only the estimates.
fit_background_fast <- function(case, exp) {
  rate <- mean(case)
  lambda_null <- -log(1 - rate) / WINDOW_DAYS
  null <- null_loglik(case)
  h_scale <- lambda_null * WINDOW_DAYS
  x <- exp / mean(exp)            # exposure with mean 1, so both parameters are near 1
  is_case <- case == 1
  fn <- function(p) {
    h <- h_scale * (p[1] + p[2] * x)
    -(sum(log(-expm1(-h[is_case]))) - sum(h[!is_case]))
  }
  gr <- function(p) {
    h <- h_scale * (p[1] + p[2] * x)
    dh <- ifelse(is_case, 1 / expm1(h), -1)
    -h_scale * c(sum(dh), sum(dh * x))
  }
  fit <- optim(c(1, 0.1), fn, gr, method = "L-BFGS-B", lower = c(1e-9, 0),
               control = list(factr = 10, pgtol = 1e-10, maxit = 500))
  if (-fit$value <= null + 1e-9 || fit$par[2] <= 0) {
    return(c(loglik = null, lambda0 = lambda_null, beta = 0))
  }
  c(loglik = -fit$value, lambda0 = fit$par[1] * lambda_null, beta = fit$par[2] * h_scale / mean(exp))
}

# With background, plus the 95% profile interval of beta and the likelihood-ratio test of beta = 0
# (one-sided, because beta >= 0: half of the chi-square with 1 df).
fit_with_background <- function(weeks) {
  case <- weeks$C
  exp <- weeks$exp
  est <- fit_background_fast(case, exp)
  null <- null_loglik(case)
  lr <- max(2 * (est[["loglik"]] - null), 0)
  profile <- function(beta) {   # best log-likelihood with beta fixed
    optimize(function(log_l0) loglik(exp(log_l0), beta, case, exp), c(-25, 5), maximum = TRUE, tol = 1e-10)$objective
  }
  above <- function(beta) profile(beta) - (est[["loglik"]] - CI_DROP)
  beta <- est[["beta"]]
  low <- if (beta == 0 || above(0) > 0) 0 else uniroot(above, c(0, beta), tol = 1e-14)$root
  high <- max(beta, 1e-12)
  while (above(high) > 0) high <- high * 2
  high <- uniroot(above, c(max(beta, high / 2), high), tol = 1e-14)$root
  data.frame(model = "with background", weeks = length(case), cases = sum(case),
             beta = beta, beta_low = low, beta_high = high, lambda0 = est[["lambda0"]],
             loglik = est[["loglik"]], aic = -2 * est[["loglik"]] + 4, p_exposure = pchisq(lr, 1, lower.tail = FALSE) / 2)
}

# ---------------------------------------------------------------- 5. delay between infection and lesion

# At-risk weeks with the window of each delay, all on the same rows (weeks with exposure for
# every delay), so the likelihoods can be compared between delays.
weeks_by_delay <- function(daily, exposure, min_healthy, delays = 0:21) {
  base <- build_weekly(daily, exposure, min_healthy = min_healthy)
  base <- base[base$S == 1, c("Agent", "date", "C")]
  windows <- lapply(delays, function(L) exposure_window(base$Agent, base$date, exposure, delay = L))
  keep <- Reduce(`&`, lapply(windows, function(w) w$exp_days > 0))
  list(weeks = base[keep, ], delays = delays, exp = sapply(windows, function(w) w$exp[keep]))
}

# Delay estimated by maximum likelihood, with its 95% profile interval. Because the delay is chosen
# by the data, the exposure test uses a permutation of the maximum over all delays (p_corrected).
delay_profile <- function(w, n_perm = 300, seed = 1) {
  case <- w$weeks$C
  null <- null_loglik(case)
  fits <- t(apply(w$exp, 2, function(e) fit_background_fast(case, e)))
  prof <- data.frame(delay = w$delays, fits, lr = 2 * (fits[, "loglik"] - null))
  best <- which.max(prof$loglik)
  inside <- prof$delay[prof$loglik >= prof$loglik[best] - CI_DROP]
  observed <- max(prof$lr)

  set.seed(seed)
  beat <- 0
  for (i in seq_len(n_perm)) {
    perm <- sample(case)
    perm_null <- null_loglik(perm)
    lr_perm <- max(apply(w$exp, 2, function(e) 2 * (fit_background_fast(perm, e)[["loglik"]] - perm_null)))
    if (lr_perm >= observed - 1e-9) beat <- beat + 1
  }
  list(profile = prof,
       summary = data.frame(weeks = length(case), cases = sum(case), best_delay = prof$delay[best],
                            delay_low = min(inside), delay_high = max(inside), beta = prof$beta[best],
                            lr = observed, p_best_delay = pchisq(observed, 1, lower.tail = FALSE) / 2,
                            p_corrected = (beat + 1) / (n_perm + 1)))
}

# Weeks of the same cow are not independent: bootstrap that resamples cows, chooses the delay
# again and re-estimates beta in each sample.
cow_bootstrap <- function(w, n_boot = 200, seed = 2) {
  cows <- unique(w$weeks$Agent)
  rows_of <- split(seq_len(nrow(w$weeks)), w$weeks$Agent)
  case <- w$weeks$C
  set.seed(seed)
  res <- t(replicate(n_boot, {
    idx <- unlist(rows_of[as.character(sample(cows, replace = TRUE))], use.names = FALSE)
    if (sum(case[idx]) == 0) return(c(NA, NA))
    fits <- apply(w$exp[idx, , drop = FALSE], 2, function(e) fit_background_fast(case[idx], e))
    best <- which.max(fits["loglik", ])
    c(w$delays[best], fits["beta", best])
  }))
  res <- res[!is.na(res[, 1]), ]
  data.frame(boot = nrow(res),
             delay_boot_low = unname(quantile(res[, 1], 0.025)), delay_boot_high = unname(quantile(res[, 1], 0.975)),
             beta_boot_low = unname(quantile(res[, 2], 0.025)), beta_boot_high = unname(quantile(res[, 2], 0.975)))
}
