#ifndef EXPANDBASIS_H
#define EXPANDBASIS_H

void expandBasis_init(double *W, int ldW, double *H, int ldH, double *P, int ldP,
                int basisSize, int dim, int numEvals, struct jdqmr16Info *jd);

void expandBasis(double *W, int ldW, double *H, int ldH, double *P, int ldP, double *Qlocked, int ldQlocked, int numLocked,
                double *AW, int ldAW, int &basisSize, int dim, int numEvals, struct jdqmr16Info *jd);

void expandBasis_destroy(struct jdqmr16Info *jd);

#endif
