functions {
  real[] sir(real t, real[] y, real[] theta, 
             real[] x_r, int[] x_i) {

      real S = y[1];
      real E = y[2];
      real I = y[3];
      real R = y[4];
      real D1 = y[5];
      real D2 = y[6];

      real N = x_i[1];
      
      real beta = theta[1];
      real gamma = theta[2];
      real cfr = theta[3];
      real sigma = theta[4];
      real lambda = theta[5];


      real dS_dt = -beta * I * S / N;
      real dE_dt =  beta * I * S / N - sigma*E;
      real dI_dt =  sigma*E - gamma * I ;
      real dR_dt =  (1-cfr)*gamma * I;
      real dD1_dt = cfr*gamma*I - lambda*D1;
      real dD2_dt = lambda*D1;

      
      return {dS_dt, dE_dt, dI_dt, dR_dt, dD1_dt,dD2_dt};
  }
}
data {
  int<lower=1> n_days;
  real y0[6];
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
  real<lower=0> sigma;
  real<lower=0> lambda;

  real<lower=0> phi_inv;
}
transformed parameters{
  real y[n_days, 6];
  real y_mean[n_days];
  real tmp[n_days,6];
  real phi = 1. / phi_inv;
  {
    real theta[5];
    theta[1] = beta[1];
    theta[2] = gamma;
    theta[3] = cfr;
    theta[4] = sigma;
    theta[5] = lambda;


  
    
  
    tmp[1:2,1:6] = integrate_ode_rk45(sir, y0, t0, ts, theta, x_r, x_i);
    y[1,1:6] = tmp[2,1:6] ;
    for(n in 2:n_days) {
        theta[1] = beta[n];
        tmp[1:2,1:6] = integrate_ode_rk45(sir, y[n-1,1:6], t0, ts, theta, x_r, x_i);
        y[n,1:6] = tmp[2,1:6] ;

    }
    
    y_mean[1] = y[1, 6];
    for(n in 2:n_days) {
      y_mean[n] = (y[n, 6] - y[n-1, 6]);
    }
  }
}
model {
  //priors
  beta[1] ~ normal(.95, .001);
  
  for (i in 2:n_days){
    beta[i] ~ normal(beta[i-1],.1);
  }
  gamma ~ normal(0.4, 0.0001);
  sigma ~ normal(0.25, 0.0001);
  lambda ~ normal(0.1, 0.0001);

  cfr ~ normal(0.01,0.01);
  phi_inv ~ exponential(1000);
  
  //sampling distribution
  //col(matrix x, int n) - The n-th column of matrix x. Here the number of infected people 
  cases ~ neg_binomial_2(y_mean,phi);
}
generated quantities {
  real R0 = beta[1] / gamma;
  real recovery_time = 1 / gamma;
  real pred_cases[n_days];
  pred_cases = neg_binomial_2_rng(y_mean,phi);
}