data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
  real<lower=0> betascale;
  int<lower=0> nu;
}

parameters {
  real<lower=0> tau;
  vector<lower=0>[H] lambda;
  vector[H] beta_raw;
  real theta_C;
  real theta_T;
}

transformed parameters{
  vector[H] theta_h;
  vector[H] beta;
  beta = tau * lambda .* beta_raw;
  theta_h = theta_C + beta;
}

model {
  target += normal_lpdf(theta_C|0,100);
  target += normal_lpdf(beta_raw|0,1);
  target += student_t_lpdf(tau|nu,0,betascale);
  target += student_t_lpdf(lambda|1,0,1);
  target += binomial_logit_lpmf(x_h|n_h,theta_h);
  target += binomial_logit_lpmf(x_CC|n_CC,theta_C);
  target += normal_lpdf(theta_T | 0, 100);
  target += binomial_logit_lpmf(x_CT|n_CT,theta_T);
}

generated quantities {
  real pi_C = inv_logit(theta_C);
  real pi_T = inv_logit(theta_T);
  real g_pi = pi_T - pi_C;
}
