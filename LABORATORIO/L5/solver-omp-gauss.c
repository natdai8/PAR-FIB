#include "omp.h"

#define lowerb(id, p, n)  ( id * (n/p) + (id < (n%p) ? id : n%p) )
#define numElem(id, p, n) ( (n/p) + (id < (n%p)) )
#define upperb(id, p, n)  ( lowerb(id, p, n) + numElem(id, p, n) - 1 )

#define min(a, b) ( (a < b) ? a : b )
#define max(a, b) ( (a > b) ? a : b )

extern int userparam;

// Function to copy one matrix into another
void copy_mat (double *u, double *v, unsigned sizex, unsigned sizey) {
    int nblocksi=omp_get_max_threads();
    int nblocksj=1;
    int blocki = omp_get_thread_num();
    int i_start = lowerb(blocki, nblocksi, sizex);
    int i_end = upperb(blocki, nblocksi, sizex);
      for (int blockj=0; blockj<nblocksj; ++blockj) {
        int j_start = lowerb(blockj, nblocksj, sizey);
        int j_end = upperb(blockj, nblocksj, sizey);
        for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++)
          for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++)
            v[i*sizey+j] = u[i*sizey+j];
  }
}

// 2D-blocked solver: one iteration step
double solve (double *u, double *unew, unsigned sizex, unsigned sizey) {
    double tmp, diff, sum=0.0;

    if(u == unew){
        int nblocksi=omp_get_max_threads();
        int next[nblocksi][16];
        int nblocksj=userparam*nblocksi;
        next[0][0] = nblocksj;
        for (int i = 1; i < nblocksi; i++) next[i][0] = 0;
        
        #pragma omp parallel private(tmp, diff) shared(next) firstprivate(nblocksi, nblocksj) reduction(+: sum)// complete data sharing constructs here
        {
        int nexttmp;
        int blocki = omp_get_thread_num();
        int i_start = lowerb(blocki, nblocksi, sizex);
        int i_end = upperb(blocki, nblocksi, sizex); 
        for (int blockj=0; blockj<nblocksj; ++blockj) {
            do {
                    #pragma omp atomic read
                    nexttmp = next[blocki][0];
            } while (nexttmp < blockj + 1); // waiting to advance to the next block
            
            int j_start = lowerb(blockj, nblocksj, sizey);
            int j_end = upperb(blockj, nblocksj, sizey);
            for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
                for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
                    tmp = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
                                u[ i*sizey	   + (j+1) ] +  // right
                                u[ (i-1)*sizey + j     ] +  // top
                                u[ (i+1)*sizey + j     ] ); // bottom
                    diff = tmp - u[i*sizey+ j];
                    sum += diff * diff;
                  #pragma omp atomic write 
                    unew[i*sizey+j] = tmp;
                }
            }
            
            if (blocki < nblocksi-1) {
                #pragma omp atomic update
                next[blocki+1][0]++;
            }
        }
        }
    }

    else{
        int nblocksi=omp_get_max_threads();
        int nblocksj=1;
        #pragma omp parallel private(tmp, diff) firstprivate(nblocksi, nblocksj) reduction(+: sum)// complete data sharing constructs here
        {
        int blocki = omp_get_thread_num();
        int i_start = lowerb(blocki, nblocksi, sizex);
        int i_end = upperb(blocki, nblocksi, sizex);
        for (int blockj=0; blockj<nblocksj; ++blockj) {
            int j_start = lowerb(blockj, nblocksj, sizey);
            int j_end = upperb(blockj, nblocksj, sizey);
            for (int i=max(1, i_start); i<=min(sizex-2, i_end); i++) {
            for (int j=max(1, j_start); j<=min(sizey-2, j_end); j++) {
                tmp = 0.25 * ( u[ i*sizey	   + (j-1) ] +  // left
                            u[ i*sizey	   + (j+1) ] +  // right
                            u[ (i-1)*sizey + j     ] +  // top
                            u[ (i+1)*sizey + j     ] ); // bottom
                diff = tmp - u[i*sizey+ j];
                sum += diff * diff;
                unew[i*sizey+j] = tmp;
            }
            }
        }
        }
    }
    return sum;
}
