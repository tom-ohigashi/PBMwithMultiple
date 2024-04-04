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
  real<lower=0, upper=1> delta[H];
  real<lower=0> a;
  real<lower=0> b;
}

transformed parameters{
  real<lower=0, upper=1> mu;
  real<lower=0, upper=1> sigmasq;
  real lik_C;
  real lik_T;
  real SC;
  real delta_sum1[H];
  real delta_sum2[H];
  real delta_sum3[H];
  real<lower=0, upper=1> pi_C = inv_logit(theta_C);
  real<lower=0, upper=1> pi_T = inv_logit(theta_T);
  
  mu = a / (a + b);
  sigmasq = (mu * (1 - mu)) / (a + b + 1);

  for(h in 1:H){
    delta_sum1[h] = delta[h] * x_h[h];
    delta_sum2[h] = delta[h] * (n_h[h] - x_h[h]);
    delta_sum3[h] = delta[h] * n_h[h];
  }
  
  lik_C = (sum(delta_sum1) + x_CC) * log(pi_C) + (sum(delta_sum2) + (n_CC - x_CC)) * log(1 - pi_C);
  lik_T = x_CT * log(pi_T) + (n_CT - x_CT) * log(1 - pi_T);
  SC = lgamma(sum(delta_sum1) + 1) + lgamma(sum(delta_sum2) + 1) - lgamma(sum(delta_sum3) + 1 + 1);
}

model {
  target += normal_lpdf(theta_C | 0, 100);
  target += normal_lpdf(theta_T | 0, 100);
  target += uniform_lpdf(mu|0,1);
  target += gamma_lpdf(1 / sigmasq|0.01,0.01);
  target += -log(sigmasq ^ 2);
  target += beta_lpdf(delta|a,b);
  target += lik_C;
  target += lik_T;
  target += -SC;
}

generated quantities {
  real g_pi = pi_T - pi_C;
}
