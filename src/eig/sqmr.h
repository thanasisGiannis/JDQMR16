#ifndef SQMR_H
#define SQMR_H

void blQmrD_init(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);

void blQmrD_destroy(blQmrSpace *spBlQmr);

void blQmrD(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);


#endif
