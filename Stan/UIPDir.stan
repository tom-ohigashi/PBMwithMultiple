data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
  vector<lower=0>[H] gamma_h;
  vector<lower=0, upper=1>[H] theta_h;
  vector<lower=0>[H] I_U;
}

parameters {
  real<lower=0, upper=1> pi_C;
  real theta_T;
  
  real<lower=0, upper=sum(n_h)> M;
  simplex[H] omega; 
}

transformed parameters{
  real<lower=0, upper=1> mu;
  real<lower=0> eta2;
  real<lower=0> alpha;
  real<lower=0> beta;

  mu = sum(omega .* theta_h);
  eta2 = 1/(M * sum(omega .* I_U));
  
  if(mu*(1-mu) > eta2){
    alpha = mu * ((mu*(1-mu)/eta2) - 1);
    beta = (1-mu) * ((mu*(1-mu)/eta2) - 1);
  }else{
    alpha = mu * 0.01;
    beta = (1-mu) * 0.01;
  }
}

model {
  target += dirichlet_lpdf(omega | gamma_h);
  target += uniform_lpdf(M | 0, sum(n_h));
  target += beta_lpdf(pi_C | alpha, beta);
  target += normal_lpdf(theta_T | 0, 100);
  target += binomial_lpmf(x_CC | n_CC, pi_C);
  target += binomial_logit_lpmf(x_CT|n_CT,theta_T);
}

generated quantities {
  real pi_T = inv_logit(theta_T);
  real g_pi = pi_T - pi_C;
  real theta_C = logit(pi_C);
}
