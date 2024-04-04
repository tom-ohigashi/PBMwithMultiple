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
  real mu;
  real<lower=0> tau;
  vector[H+1] eta_raw;
  real theta_T;
}

transformed parameters{
  real theta_C;
  vector[H] theta_h;

  for(h in 1:H){
    theta_h[h] = mu + tau*eta_raw[h];
  }
  theta_C = mu + tau*eta_raw[H+1];
}

model {
  target += normal_lpdf(mu|0, 100);
  target += normal_lpdf(tau|0, 1);
  target += normal_lpdf(eta_raw|0,1);
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
