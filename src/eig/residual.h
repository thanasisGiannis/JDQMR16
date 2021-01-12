#ifndef RESIDUAL_H
#define RESIDUAL_H


#if 0
void residual(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd);

void residual_init(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd);
void residual_destroy(struct jdqmr16Info *jd);
#else
void residual_init(double *R, int ldR, double *V, int ldV, double *L, int numEvals, struct jdqmr16Info *jd);
void residual_destroy(struct jdqmr16Info *jd);


void residual(double *R, int ldR, double *V, int ldV, double *L, double *AV, int ldAV, double *QH, int ldQH,
               int numEvals, int basisSize, struct jdqmr16Info *jd);

#endif
#endif

