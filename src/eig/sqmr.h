#ifndef SQMR_H
#define SQMR_H

void sqmr_init(half *X, int ldX, half *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd);
void sqmr(half *X, int ldX, half *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd);
void sqmr_destroy(struct jdqmr16Info *jd);



void sqmrF_init(float *X, int ldX, float *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd);
void sqmrF(float *X, int ldX, float *B, int ldB, int dim, double infNormB, struct jdqmr16Info *jd);
void sqmrF_destroy(struct jdqmr16Info *jd);



void sqmrD_init(double *X, int ldX, double *B, int ldB, double *V,int ldV, int numEvals,
               int dim, double infNormB, struct jdqmr16Info *jd);

void sqmrD(double *X, int ldX, double *B, int ldB, double *V,int ldV, int numEvals,
            int dim, double infNormB, struct jdqmr16Info *jd);

void sqmrD_destroy(struct jdqmr16Info *jd);

#endif
