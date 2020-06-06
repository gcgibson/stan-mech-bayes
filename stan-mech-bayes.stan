functions {
  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real S = y[1];
      real I = y[2];
      real R = y[3];
      real D = y[4];
      real N = x_i[1];
      
      real beta = theta[1];
      real gamma = theta[2];
      real cfr = theta[3];

      real dS_dt = -beta * I * S / N;
      real dI_dt =  beta * I * S / N - gamma * I - cfr*I;
      real dR_dt =  gamma * I;
      real dD_dt = cfr*I;
      
      return {dS_dt, dI_dt, dR_dt, dD_dt};
  }
}
data {
  int<lower=1> n_days;
  real y0[4];
  real t0;
  real ts[2];
  int N;
  int cases[n_days];
}
transformed data {
  real x_r[0];
  int x_i[1] = { N };
}
parameters {
  real<lower=0> gamma;
  real<lower=0> beta[n_days];
  real<lower=0> cfr;

  real<lower=0> phi_inv;
}
transformed parameters{
  real y[n_days, 4];
  real y_mean[n_days];
  real tmp[n_days,4];
  real phi = 1. / phi_inv;
  {
    real theta[3];
    theta[1] = beta[1];
    theta[2] = gamma;
    theta[3] = cfr;
  
    
  
    tmp[1:2,1:4] = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i);
    y[1,1:4] = tmp[2,1:4] ;
    for(n in 2:n_days) {
        theta[1] = beta[n];
        tmp[1:2,1:4] = integrate_ode_rk45(sir, y[n-1,1:4], t0, ts, theta, x_r, x_i);
        y[n,1:4] = tmp[2,1:4] ;

    }
    
    y_mean[1] = y[1, 4];
    for(n in 2:n_days) {
      y_mean[n] = (y[n, 4] - y[n-1, 4]);
    }
  }
}
model {
  //priors
  beta[1] ~ normal(.95, .01);
  
  for (i in 2:n_days){
    beta[i] ~ normal(beta[i-1],.01);
  }
  gamma ~ normal(0.4, 0.01);
  cfr ~ normal(0.01,0.01);
  phi_inv ~ exponential(1);
  
  //sampling distribution
  //col(matrix x, int n) - The n-th column of matrix x. Here the number of infected people 
  cases ~ neg_binomial_2(y_mean, phi);
}
generated quantities {
  real R0 = beta[1] / gamma;
  real recovery_time = 1 / gamma;
  real pred_cases[n_days];
  pred_cases = neg_binomial_2_rng(y_mean, phi);
}