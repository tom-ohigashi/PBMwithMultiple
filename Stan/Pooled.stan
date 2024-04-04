data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
}

parameters {
  real theta_C;
  real theta_T;
}

model {
  target += normal_lpdf(theta_C | 0, 100);
  target += normal_lpdf(theta_T | 0, 100);
  target += binomial_logit_lpmf(x_h |n_h ,theta_C);
  target += binomial_logit_lpmf(x_CC|n_CC,theta_C);
  target += binomial_logit_lpmf(x_CT|n_CT,theta_T);
}

generated quantities {
  real pi_T = inv_logit(theta_T);
  real pi_C = inv_logit(theta_C);
  real g_pi = pi_T - pi_C;
}
