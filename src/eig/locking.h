#ifndef LOCKING_H
#define LOCKING_H

void lock_init(double *V, int ldV, double *L, double *R, int ldR, double *normr,
            double *Qlocked, int ldQlocked, double *Llocked, double *W, int ldW, double *H, int ldH, double *AW, int ldAW, 
            int &numLocked, int &numEvals, int maxBasis, int &basisSize, int dim, double tol, struct jdqmr16Info *jd);

void lock_destroy(struct jdqmr16Info *jd);


void lock(double *V, int ldV, double *L, double *R, int ldR, double *normr,
            double *Qlocked, int ldQlocked, double *Llocked, double *W, int ldW, double *H, int ldH, double *AW, int ldAW, 
            int &numLocked, int &numEvals, int maxBasis, int &basisSize, int dim, double tol, struct jdqmr16Info *jd);

#endif
