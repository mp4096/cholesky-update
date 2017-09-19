// Compute the downdate of an upper Cholesky decomposition.
//
// This function is intended to be used as an (almost) drop-in replacement
// for MATLAB's `cholupdate` if code generation is required.
//
//
// `cholesky_downdate_real` is based on LINPACK subroutine `DCHDD`.
// However, the (z,y,rho)-downdate is not implemented.
//
// `euclidean_norm_real` is based on LAPACK/BLAS subroutine `DNRM2`.
//
// Important: This program requires C99! See the compilers note below.
//
//
// Copyright (c) 2011 University of Tennessee.
//
// Copyright (c) 2011 University of California Berkeley.
//
// Copyright (c) 2011 University of Colorado Denver.
//
// Copyright (c) 2011 NAG Ltd.
//
// Copyright (c) 2011 Sven Hammarling,
//                    NAG Ltd.
//
// Copyright (c) 1978, G. W. Stewart,
//                     University of Maryland, Argonne National Lab.
//
// Copyright (c) 2016-2017, Mikhail Pak,
//                     Technical University of Munich.
//
// Notice on compilers:
// Microsoft Windows SDK 7.1 does not support C99.
// As a result, a lot of required functions are missing in `math.h`.
//
// Following compilers have been tested:
//   - MinGW 4.9.2 C/C++ (TDM-GCC)
//   - Microsoft Visual C++ 2015 Professional
//
// Compile this function by typing `mex CholeskyDowndateReal.c`


#include "matrix.h" // mwIndex, mwSize
#include "mex.h" // MEX functions and types
#include <math.h> // sqrt, fabs, hypot


// Define functions
// Rank-1 Cholesky downdate
int cholesky_downdate_real(const mwSize n, double *R, const double *x);
// Dot product of two real vectors
double dot_product_real(const mwIndex size, const double *x, const double *y);
// Euclidean norm of a real vector
double euclidean_norm_real(const mwIndex size, const double *x);


// Entry point for the MEX interface
void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs) {
    // Pointer to the status output
    double *status_out = NULL;
    // Exit on error?
    // If `true`, then throw an error if the matrix could not be downdated.
    // If `false`, continue on error, but supply a status output.
    bool exit_on_error;

    // Check if the number of input arguments is OK
    if (nrhs != 2) {
        mexErrMsgIdAndTxt(
            "CholeskyDowndateReal:InvalidInput",
            "Invalid number of input arguments. Please supply R and x.");
        return;
    }

    // Check the number of output arguments
    if (nlhs == 1) {
        exit_on_error = true;
    } else if (nlhs == 2) {
        exit_on_error = false;
    } else {
        mexErrMsgIdAndTxt(
            "CholeskyDowndateReal:InvalidCall",
            "Invalid number of output arguments: only 1 or 2 allowed.");
        return;
    }

    // Perform a deep copy of the first input (matrix `R`).
    // Unfortunately, in-place update of `R` is not possible due to the
    // copy-on-write nature of MATLAB. We have to copy `R` and modified the
    // copied array in-place.
    plhs[0] = mxDuplicateArray(prhs[0]);
    // Target the `R_upd` pointer to the freshly created output value (downdated Cholesky factor)
    double *R_upd = mxGetPr(plhs[0]);

    // Target the `x` pointer to the second input argument (vector `x`)
    const double *x = mxGetPr(prhs[1]);

    if (!exit_on_error) {
        // Target the `status_out` pointer to the second output value
        plhs[1] = mxCreateDoubleScalar(mxREAL);
        // Target the `status_out` pointer to the second output value
        status_out = mxGetPr(plhs[1]);
    }

    // Get size of the inputs
    const mwSize *size_R = mxGetDimensions(prhs[0]);
    const mwSize *size_x = mxGetDimensions(prhs[1]);

    // Check if R is square
    if (size_R[0] != size_R[1]) {
        mexErrMsgIdAndTxt("CholeskyDowndateReal:InvalidInput",
                          "R must be a square matrix.");
        return;
    }
    // Check if x is a column vector
    if (size_x[1] != 1) {
        mexErrMsgIdAndTxt("CholeskyDowndateReal:InvalidInput",
                          "x must be a column vector.");
        return;
    }
    // Check if dimensions of the R and x are consistent
    if (size_R[1] != size_x[0]) {
        mexErrMsgIdAndTxt("CholeskyDowndateReal:InvalidInput",
                          "R and x must have the same size.");
        return;
    }

    // Call the Cholesky downdate
    int status = cholesky_downdate_real(size_R[0], R_upd, x);

    if (status == 0) {
        // Everything OK
        return;
    } else if (status == -1) {
        if (exit_on_error) {
            // Throw an error and exit
            mexErrMsgIdAndTxt("CholeskyDowndateReal:downdatedMatrixNotPosDef",
                              "Downdated matrix must be positive definite.");
            return;
        } else {
            // Supply the error status and return.
            // Returned `R_upd` is equal to `R`.
            *status_out = (double)status;
            return;
        }
    } else if (status == 1) {
        if (exit_on_error) {
            // Throw an error and exit
            mexErrMsgIdAndTxt("CholeskyDowndateReal:AllocateError",
                              "Could not allocate memory.");
            return;
        } else {
            // Supply the error status and return.
            // Returned `R_upd` is equal to `R`.
            *status_out = (double)status;
            return;
        }
    } else {
        // This _cannot_ happen.
        mexErrMsgIdAndTxt("CholeskyDowndateReal:UnknownError",
                          "Something very bad happened.");
        return;
    }
}


int cholesky_downdate_real(const mwSize n, double *R, const double *x) {
    // This function returns:
    //  0 if successful;
    // -1 if the downdated matrix is not positive definite;
    //  1 if could not allocate memory for temp arrays;

    // Allocate memory for the vectors with the cosines of transforming rotations
    double *c = mxMalloc(n * sizeof(double));
    if (c == NULL) {
        return 1;
    }
    // Allocate memory for the vectors with the sines of transforming rotations
    double *s = mxMalloc(n * sizeof(double));
    if (s == NULL) {
        return 1;
    }

    // Solve the system R^T*a = x, placing the result in the vector `s`

    // Solve for the first element
    // `*(r + n*0 + 0)` is simply the matrix entry R(1, 1)
    s[0] = x[0] / (*(R + n * 0 + 0));

    for (mwIndex j = 1; j < n; j++) {
        s[j] = x[j] - dot_product_real(j, (R + j * n), s);
        s[j] /= *(R + n * j + j);
    }

    const double norm = euclidean_norm_real(n, s);
    if (norm >= 1.0) {
        // The downdated matrix is not positive definite
        return -1;
    }
    double alpha = sqrt(1.0 - norm * norm);

    // Determine the transformations
    for (mwIndex j = 0; j < n; j++) {
        const mwIndex i = n - 1 - j;
        const double scale = alpha + fabs(s[i]);
        const double a = alpha / scale;
        const double b = s[i] / scale;
        const double r = hypot(a, b);
        c[i] = a / r;
        s[i] = b / r;
        alpha = scale * r;
    }

    // Apply the transformations to r
    for (mwIndex j = 0; j < n; j++) {
        double xx = 0.0;
        for (mwIndex k = 0; k <= j; k++) {
            const mwIndex i = j - k;
            // Target R_ij to the matrix entry R(i + 1, j + 1)
            // IMPORTANT:
            // R is in the column major notation!
            // (since it was copied from MATLAB)
            double *R_ij = R + j * n + i;

            const double t = xx * c[i] + (*R_ij) * s[i];
            *R_ij = (*R_ij) * c[i] - xx * s[i];
            xx = t;
        }
    }

    // Free memory
    mxFree(c);
    mxFree(s);

    // Everything OK
    return 0;
}


double dot_product_real(const mwIndex size, const double *x, const double *y) {
    // Initialise return value for the dot product
    double res = 0.0;
    for (mwIndex i = 0; i < size; i++) {
        res += x[i] * y[i];
    }
    return res;
}


double euclidean_norm_real(const mwIndex size, const double *x) {
    // Handle trivial cases
    if (size < 1) {
        return 0.0;
    }
    if (size == 0) {
        return fabs(x[0]);
    }

    // Initialise temp variables
    double scale = 0.0;
    double ssq = 1.0;

    // Inlined `DLASSQ`
    for (mwIndex i = 0; i < size; i++) {
        if (x[i] != 0.0) {
            const double abs_curr_x = fabs(x[i]);

            if (scale < abs_curr_x) {
                ssq = 1.0 + ssq * (scale / abs_curr_x) * (scale / abs_curr_x);
                scale = abs_curr_x;
            } else {
                ssq += (abs_curr_x / scale) * (abs_curr_x / scale);
            }
        }
    }

    return scale * sqrt(ssq);
}
