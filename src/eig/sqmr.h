#ifndef SQMR_H
#define SQMR_H

void blQmrD_init(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);

void blQmrD_destroy(blQmrSpace *spBlQmr);

void blQmrD(double *X, int ldX, double *B, int ldB, double *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx, double tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);

void blQmrF(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx,  float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);


void blQmrF_init(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);

void blQmrF_destroy(blQmrSpace *spBlQmr);


void blQmrH(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, int pivotThitaIdx, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);


void blQmrH_init(float *X, int ldX, float *B, int ldB, float *Q, int ldQ,
             int dim, int numEvals, int maxNumEvals, float tol, int maxIter,
             struct blQmrSpace *spBlQmr, struct jdqmr16Info *jd);

void blQmrH_destroy(blQmrSpace *spBlQmr);


#endif
