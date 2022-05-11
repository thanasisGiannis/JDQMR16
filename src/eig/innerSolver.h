#ifndef INNERSOLVER_H
#define INNERSOLVER_H


void innerSolver(double *X, int ldX, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd);
void innerSolver_destroy(struct jdqmr16Info *jd);

void innerSolver_init(double *X, int ldX, double *R, int ldR, 
                  double *V, int ldV, double *L,
                  int numEvals, int dim, struct jdqmr16Info *jd);


#endif
