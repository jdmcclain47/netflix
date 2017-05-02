#include <stdio.h>

void grad_U(int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    double diff = 0.0;
    diff += Yij;
    for (int i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    for (int i = 0; i < k; ++i){
        Ui[i] = (1-reg*eta)*Ui[i] + eta*Vj[i]*diff;
    }
}

void grad_V(int k, double* Ui, double Yij, double* Vj, double reg, double eta){
    double diff = 0.0;
    diff += Yij;
    for (int i = 0; i < k; ++i){
        diff -= Ui[i]*Vj[i];
    }
    for (int i = 0; i < k; ++i){
        Vj[i] = (1-reg*eta)*Vj[i] + eta*Ui[i]*diff;
    }
}

double c_multiply(int n_samples, 
                  long* u, 
                  long* v, 
                  double* r,
                  double* Umat,
                  double* Vmat,
                  int m,
                  int n,
                  int k,
                  double reg,
                  double eta
                  ){
    double result = 0.0;
    long user, movie;
    double Yij;
    double* Ui;
    double* Vj;
    for (int i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        Yij = r[i];
        Ui = &Umat[user*k];
        Vj = &Vmat[movie*k];
        grad_U(k,Ui,Yij,Vj,reg,eta);
        Ui = &Umat[user*k];
        Vj = &Vmat[movie*k];
        grad_V(k,Ui,Yij,Vj,reg,eta);
    }
    return result;
}

double c_err(int n_samples,
             long* u, 
             long* v, 
             double* r,
             double* Umat,
             double* Vmat,
             int m,
             int n,
             int k
        ){
    double err = 0.0;
    double uvdot;
    long user, movie;
    double Yij;

    for (int i = 0; i < n_samples; ++i){
        user   = u[i];
        movie  = v[i];
        Yij = r[i];
        uvdot = 0.0 ;
        for (int j = 0; j < k; ++j){
            uvdot += Umat[user*k + j]*Vmat[movie*k + j];
        }
        uvdot -= Yij;
        uvdot *= uvdot;
        err += uvdot;
    }
    err *= 0.5;
    return err;
}

