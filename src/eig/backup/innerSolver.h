#ifndef INNERSOLVER_H
#define INNERSOLVER_H


void innerSolver(double *P, int ldP, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd);
void innerSolver_destroy(struct jdqmr16Info *jd);

void innerSolver_init(double *P, int ldP, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd);


#endif
