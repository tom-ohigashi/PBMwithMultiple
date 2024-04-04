data {
  int<lower=0> H;
  int<lower=0> x_h[H];
  int<lower=0> n_h[H];
  int<lower=0> x_CC;
  int<lower=0> n_CC;
  int<lower=0> x_CT;
  int<lower=0> n_CT;
  int C;
}

parameters {
  real mu;
  vector<lower=0>[H+1] delta;
  vector<lower=0, upper=1>[C] q;
  vector<lower=0>[C] sigma_mix;
  real theta_T;
}

transformed parameters{
  real theta_C;
  vector[H] theta_h;

  vector<lower=0>[C] p;
  simplex[C] w_c;

  for(h in 1:H){
    theta_h[h] = mu + delta[h];
  }
  theta_C = mu + delta[H+1];

   p[1] = q[1];
   for (j in 2:C)
      p[j] = q[j]*(1 - q[j-1])*p[j-1]/q[j-1];
   for (j in 1:C)
      w_c[j] = p[j]/sum(p);
}

model {
  for (n in 1:H+1) {
      real ps[C];
      for (j in 1:C)
         ps[j] = log(w_c[j]) + normal_lpdf(delta[n]| 0, sigma_mix[j]);
      target += log_sum_exp(ps);
  }
  q ~ beta(1, 1);

  target += normal_lpdf(sigma_mix|0,1);
  target += normal_lpdf(mu|0, 100);
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
