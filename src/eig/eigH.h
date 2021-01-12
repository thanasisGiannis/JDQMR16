#ifndef EIGH_H
#define EIGH_H


void eigH_init(double *W, int ldW, double *L, double *H, int ldH, int numEvals, int maxBasisSize, struct jdqmr16Info *jd);

void eigH(double *V, int ldV, double *L, double *W, int ldW,
          double *H, int ldH, int numEvals, int basisSize, double* QH,int ldQH, struct jdqmr16Info *jd);

void eigH_destroy(struct jdqmr16Info *jd);

#endif
