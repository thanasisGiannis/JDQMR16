#ifndef RESIDUAL_H
#define RESIDUAL_H


void residual(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd);

void residual_init(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd);
void residual_destroy(struct jdqmr16Info *jd);

#endif

