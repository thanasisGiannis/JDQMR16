#ifndef INITBASIS_H
#define INITBASIS_H


void initBasis_init(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd);

void initBasis(double *W, int ldW, double *H, int ldH, double *V, int ldV, double *L, double *AW, int ldAW,
                int dim, int maxSizeW, int numEvals, struct jdqmr16Info *jd);

void initBasis_destroy(struct jdqmr16Info *jd);

#endif
