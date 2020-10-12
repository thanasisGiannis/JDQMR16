#ifndef SQMR_H
#define SQMR_H


void sqmr(half *X, int ldX, half *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd);

#endif
