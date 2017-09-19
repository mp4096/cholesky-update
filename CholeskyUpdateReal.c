/*
* Compute the update of an upper Cholesky decomposition.
*
* This function is intended to be used as an (almost) drop-in replacement
* for MATLAB's `cholupdate` if code generation is required.
*
*
* `cholesky_update_real` is based on LINPACK subroutine `DCHUD`.
* However, the (z,y,rho)-update is not implemented.
*
* `givens_rotation_real` is based on LAPACK/BLAS subroutine `DROTG`.
*
*
* Copyright (c) 2011 University of Tennessee
*
* Copyright (c) 2011 University of California Berkeley
*
* Copyright (c) 2011 University of Colorado Denver
*
* Copyright (c) 2011 NAG Ltd.
*
* Copyright (c) 2011 Sven Hammarling,
*                    NAG Ltd.
*
* Copyright (c) 1978, G. W. Stewart,
*                     University of Maryland, Argonne National Lab.
*
* Copyright (c) 2016, Mikhail Pak,
*                     Technical University of Munich.
*/

/*
* Notice on compiler portability:
* Microsoft Windows SDK 7.1 seems to have problems with `math.h`.
* Following compilers have been tested:
*   - MinGW 4.9.2 C/C++ (TDM-GCC)
*   - Microsoft Visual C++ 2015 Professional
*
* Compile this function by typing `mex CholeskyUpdateReal.c`
*/


#include "matrix.h" /* mwIndex, mwSize */
#include "mex.h"    /* MEX functions and types */
#include <math.h>   /* sqrt, fabs, hypot */


/* Define functions */
/* Rank-1 Cholesky update */
int cholesky_update_real(const mwSize n, double *R, double *x);
/* Real Givens rotation */
void givens_rotation_real(double *a, double *b, double *c, double *s);


/* Entry point for the MEX interface */
void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    /* Pointer to the copied and updated Cholesky factor */
    double *R_upd = NULL;
    /* Pointer to the input vector `x` */
    double *x = NULL;
    /* Arrays for the size of input arguments */
    const mwSize *size_x, *size_R;
    /* Return code of the Cholesky update */
    int retval;

    /* Check if the number of input arguments is OK */
    if (nrhs != 2) {
        mexErrMsgIdAndTxt(
            "CholeskyUpdateReal:InvalidInput",
            "Invalid number of input arguments. Please supply R and x.");
        return;
    }

    /* Check the number of output arguments */
    if (nlhs != 1) {
        mexErrMsgIdAndTxt(
            "CholeskyUpdateReal:InvalidCall",
            "Invalid number of output arguments: only 1 or 2 allowed.");
        return;
    }

    /*
    * Perform a deep copy of the first input (matrix `R`).
    * Unfortunately, in-place update of `R` is not possible due to the
    * copy-on-write nature of MATLAB. We have to copy `R` and modified the
    * copied array in-place.
    */
    plhs[0] = mxDuplicateArray(prhs[0]);
    /* Target the `R_upd` pointer to the freshly created output value */
    R_upd = mxGetPr(plhs[0]);

    /* Target the `x` pointer to the second input argument */
    x = mxGetPr(prhs[1]);

    /* Get size of the inputs*/
    size_R = mxGetDimensions(prhs[0]);
    size_x = mxGetDimensions(prhs[1]);

    /* Check if R is square */
    if (size_R[0] != size_R[1]) {
        mexErrMsgIdAndTxt("CholeskyUpdateReal:InvalidInput",
                          "R must be a square matrix.");
        return;
    }
    /* Check if x is a column vector */
    if (size_x[1] != 1) {
        mexErrMsgIdAndTxt("CholeskyUpdateReal:InvalidInput",
                          "x must be a column vector.");
        return;
    }
    /* Check if dimensions of the R and x are consistent */
    if (size_R[1] != size_x[0]) {
        mexErrMsgIdAndTxt("CholeskyUpdateReal:InvalidInput",
                          "R and x must have the same size.");
        return;
    }

    /* Call the Cholesky update */
    retval = cholesky_update_real(size_R[0], R_upd, x);
    if (retval == 0) {
        /* Everything ok */
        return;
    } else if (retval == 1) {
        mexErrMsgIdAndTxt("CholeskyUpdateReal:AllocateError",
                          "Could not allocate memory.");
        return;
    } else {
        mexErrMsgIdAndTxt("CholeskyUpdateReal:UnknownError",
                          "Unknown error occurred.");
        return;
    }
}


int cholesky_update_real(const mwSize n, double *R, const double *x) {
    /* Vector with cosines of the transforming rotations */
    double *c = NULL;
    /* Vector with sines of the transforming rotations */
    double *s = NULL;
    /* Pointer to a specific matrix entry -- just a shortcut */
    double *R_ij = NULL;

    /* Intermediate variables for the update algorithm */
    double xj, t;

    /* for-loop counters */
    mwIndex i, j;

    /* Allocate memory for the vectors with sines and cosines */
    c = (double *)mxMalloc(n * sizeof(double));
    if (c == NULL) {
        return 1;
    }
    s = (double *)mxMalloc(n * sizeof(double));
    if (s == NULL) {
        return 1;
    }

    /* Update `R` */
    for (j = 0; j < n; j++) {
        xj = x[j];

        /* Apply the previous rotations */
        for (i = 0; i < j; i++) {
            /*
            * Target R_ij to the matrix entry R(i + 1, j + 1)
            * IMPORTANT:
            * R is in the column major notation!
            * (since it was copied from MATLAB)
            */
            R_ij = R + j * n + i;

            t = (*R_ij) * c[i] + xj * s[i];
            xj = xj * c[i] - (*R_ij) * s[i];
            *R_ij = t;
        }

        /* Compute the next rotation */
        givens_rotation_real((R + j * n + j), &xj, (c + j), (s + j));
    }

    /* Free memory */
    mxFree(c);
    mxFree(s);

    return 0;
}


void givens_rotation_real(double *a, double *b, double *c, double *s) {
    double r, z, roe, scale;

    if (fabs(*a) > fabs(*b)) {
        roe = *a;
    } else {
        roe = *b;
    }

    scale = fabs(*a) + fabs(*b);

    if (scale == 0.0) {
        *c = 1.0;
        *s = 0.0;
        r = 0.0;
        z = 0.0;
    } else {
        r = scale * hypot((*a / scale), (*b / scale));
        r = copysign(r, roe);

        *c = (*a) / r;
        *s = (*b) / r;

        z = 1.0;
        if (fabs(*a) > fabs(*b)) {
            z = *s;
        }
        if ((fabs(*a) >= fabs(*b)) && (*c != 0.0)) {
            z = 1.0 / (*c);
        }
    }

    *a = r;
    *b = z;
}
