data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
  real<lower=0> lambda_spike;
  real<lower=0> lambda_slab;
}

parameters {
  vector<lower=0, upper=1>[H] w;
  real theta_C;
  real theta_T;
  vector[H] beta;
}

transformed parameters{
  vector[H] theta_h;
  theta_h = theta_C + beta;
}

model {
  target += beta_lpdf(w | 1, 1);
  for(j in 1:H)
    target += log_sum_exp(log(w[j]) + double_exponential_lpdf(beta[j] | 0, lambda_spike), log1m(w[j]) + double_exponential_lpdf(beta[j] | 0, lambda_slab));
  target += normal_lpdf(theta_C|0,100);
  target += binomial_logit_lpmf(x_h|n_h,theta_h);
  target += binomial_logit_lpmf(x_CC|n_CC,theta_C);
  target += normal_lpdf(theta_T|0,100);
  target += binomial_logit_lpmf(x_CT|n_CT,theta_T);
}

generated quantities {
  real pi_C = inv_logit(theta_C);
  real pi_T = inv_logit(theta_T);
  real g_pi = pi_T - pi_C;
}
