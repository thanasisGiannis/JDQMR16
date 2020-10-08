#ifndef RESTART_H
#define RESTART_H


void restart_init(double *W, int ldW, double *H, int ldH, 
               double *Vprev, int ldVprev, double *Lprev,
               double *V, int ldV, double *L,
               int *basisSize, int maxBasisSize, int numEvals, int dim, 
               struct jdqmr16Info *jd);

void restart_destroy(struct jdqmr16Info *jd);


void restart(double *W, int ldW, double *H, int ldH, 
               double *Vprev, int ldVprev, double *Lprev,
               double *V, int ldV, double *L, double *AWp, int ldAWp,
               int *basisSize, int maxBasisSize, int numEvals, int dim, struct jdqmr16Info *jd);

#endif

