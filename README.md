# Rank-1 update and downdate of Cholesky factorization

Implemented in C99 and tested against MATLAB implementation.

## Prerequisites

Following compilers have been tested:
- MinGW 4.9.2 C/C++ (TDM-GCC)
- Microsoft Visual C++ 2015 Professional

:warning: Since these functions are implemented in C99,
you cannot use Microsoft Windows SDK 7.1.


## How to compile

To compile, type

```sh
mex CFLAGS="$CFLAGS -std=c99" CholeskyUpdateReal.c
mex CFLAGS="$CFLAGS -std=c99" CholeskyDowndateReal.c
```

Or, if you don't need portability:

```sh
mex CFLAGS="$CFLAGS -std=c99 -o3 -march=native" CholeskyUpdateReal.c
mex CFLAGS="$CFLAGS -std=c99 -o3 -march=native" CholeskyDowndateReal.c
```

## But why?

These functions were translated from FORTRAN many æons ago
since Simulink code generator didn't support MATLAB's `cholupdate` then.

However, it seems that the current (September 2017) MATLAB Coder version
supports `cholupdate`, so I guess you'll need this implementation only
if you don't have the latest MATLAB version.
