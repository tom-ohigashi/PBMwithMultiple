data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
  real<lower=0> a;
}

parameters {
  real<lower=0> tau;
  vector<lower=0>[H] psi;
  simplex[H] phi;
  real theta_C;
  real theta_T;
  vector[H] beta_raw;
}

transformed parameters{
  vector[H] theta_h;
  vector[H] beta;
  for(j in 1:H){
    beta[j] = psi[j] * (phi[j]^2) * (tau^2) * beta_raw[j];
  }

  theta_h = theta_C + beta;
}

model {
  target += normal_lpdf(theta_C|0,100);
  target += normal_lpdf(beta_raw|0,1);
  target += exponential_lpdf(psi | 0.5);
  target += gamma_lpdf(tau | H*a, 0.5);
  phi ~ dirichlet(rep_vector(a, H));
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
