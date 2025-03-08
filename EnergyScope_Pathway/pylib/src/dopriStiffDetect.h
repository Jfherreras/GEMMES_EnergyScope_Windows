#ifndef dopriStiff_h
#define dopriStiff_h

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <vector>  // <-- ADDED FOR std::vector

#if UseRCPP==1
  #include "preprocRCPP_R.h"
#else
  #include "preproc.h"
#endif
#include "ODE.h"

// TO DO: ...
using namespace std;

template<typename T>
class dopriStiff : virtual public ODE<T> {
public:
    // =====================
    //  CONSTRUCTORS
    // =====================
    dopriStiff()
      : atol(1e-4), rtol(0),
        fac(0.85), facMin(0.1), facMax(3),
        nStepMax(1000),
        hInit(0.01), hMin(1e-4), hMax(0.1),
        nStiffMax(999999), nStiffSuccessiveMax(5)
    {}

    dopriStiff(T hInitIn, T hMinIn, T hMaxIn)
      : atol(1e-4), rtol(0),
        fac(0.85), facMin(0.1), facMax(3),
        nStepMax(1000),
        hInit(hInitIn), hMin(hMinIn), hMax(hMaxIn),
        nStiffMax(999999), nStiffSuccessiveMax(5)
    {}

    dopriStiff(T atolIn, T rtolIn, T facIn, T facMinIn, T facMaxIn,
               int nStepMaxIn,
               T hInitIn, T hMinIn, T hMaxIn,
               int nStiffMaxIn, int nStiffSuccessiveMaxIn)
      : atol(atolIn), rtol(rtolIn),
        fac(facIn), facMin(facMinIn), facMax(facMaxIn),
        nStepMax(nStepMaxIn),
        hInit(hInitIn), hMin(hMinIn), hMax(hMaxIn),
        nStiffMax(nStiffMaxIn), nStiffSuccessiveMax(nStiffSuccessiveMaxIn)
    {}

    using ODE<T>::Func;
    using ODE<T>::makeEventVar;
    using ODE<T>::makeEventTime;
    using ODE<T>::completeOut;
    using ODE<T>::getNV;
    using ODE<T>::getNIV;
    using ODE<T>::getNt;
    using ODE<T>::getNRowOut;
    using ODE<T>::getTInit;
    using ODE<T>::getTEnd;
    using ODE<T>::getHOut;

    // =====================
    //  SOLVE (FULL TRAJECTORY)
    // =====================
    void solve(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        const int nRowOut = getNRowOut();
        // We'll store how many times we've detected stiffness in stiffCompt
        int stiffCompt[2] = {0, 0};

        T t = getTInit();
        T h = hInit;

        // Replace old T k0[nV], etc. with vectors
        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV), k4(nV), k5(nV), k6(nV), k7(nV);
        std::vector<T> x(nIV);
        std::vector<T> yTemp(nV), y0(nV), y4thOrder(nV);
        std::vector<T> tol(nV);

        // Initialize the first row of output
        for (int it = 0; it < nV; it++) {
            out[it] = yInit[it];
        }

        // MAIN LOOP
        for (int it = 1; it < nt; it++) {
        #if UseEventTime
            makeEventTime(t, parms, &out[(it - 1) * nRowOut], x.data(), hInit);
        #endif
        #if UseEventVar
            makeEventVar(t, parms, &out[(it - 1) * nRowOut], x.data(), hInit);
        #endif

            // One dopri step
            dopriOneStep(t,
                         &out[(it - 1) * nRowOut],
                         parms,
                         k0.data(), k1.data(), k2.data(), k3.data(),
                         k4.data(), k5.data(), k6.data(), k7.data(),
                         x.data(),
                         &out[it * nRowOut],
                         h, tol.data(), yTemp.data(), y0.data(),
                         y4thOrder.data(), stiffCompt);

            // completeOut with ydot=k0 at time t
            completeOut(k0.data(), x.data(), &out[(it - 1) * nRowOut]);
        }

        // Complete out at last point
        Func(t, &out[(nt - 1) * nRowOut], parms, k0.data(), x.data());
        completeOut(k0.data(), x.data(), &out[(nt - 1) * nRowOut]);
    }

    // =====================
    //  SOLVE LAST POINT ONLY
    // =====================
    void solveLastPoint(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();

        int stiffCompt[2] = {0, 0};
        T t = getTInit();
        T h = hInit;

        // Replace arrays with vectors
        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV),
                       k4(nV), k5(nV), k6(nV), k7(nV);
        std::vector<T> x(nIV);
        std::vector<T> y(nV), yTemp(nV), y0(nV), y4thOrder(nV);
        std::vector<T> tol(nV);

        // Copy yInit into y
        for (int it = 0; it < nV; it++) {
            y[it] = yInit[it];
        }

        // MAIN LOOP
        for (int step = 1; step < nt; step++) {
        #if UseEventTime
            makeEventTime(t, parms, y.data(), x.data(), hInit);
        #endif
        #if UseEventVar
            makeEventVar(t, parms, y.data(), x.data(), hInit);
        #endif

            // dopri step: writing new state back into y
            dopriOneStep(t,
                         y.data(),
                         parms,
                         k0.data(), k1.data(), k2.data(), k3.data(),
                         k4.data(), k5.data(), k6.data(), k7.data(),
                         x.data(),
                         y.data(),
                         h, tol.data(), yTemp.data(), y0.data(),
                         y4thOrder.data(), stiffCompt);
        }

        // final copy
        for (int it = 0; it < nV; it++) {
            out[it] = y[it];
        }
        // final derivative
        Func(t, y.data(), parms, k0.data(), x.data());
        completeOut(k0.data(), x.data(), out);
    }

protected:
    // ================
    // Single DOPRI Step
    // ================
    void dopriOneStep(
        T& t,
        const T* y,     // current state
        const T* parms,
        T* k0, T* k1, T* k2, T* k3,
        T* k4, T* k5, T* k6, T* k7,
        T* x,
        T* out,         // new state
        T& h,
        T* tol,
        T* yTemp,
        T* y0,
        T* y4thOrder,
        int* stiffCompt
    ) {
        const int nV = this->getNV();
        T tEndOneStep = t + this->getHOut();
        int nStep = 0;
        T error = 0.0;
        T alpha = 0.0;
        T hMem = 0.0;
        T lambda = 0.0;
        T lambdaNum = 0.0;
        T lambdaDenom = 0.0;
        bool justRejected = false;
        bool truncatedH = false;

        // We'll need a local array for g6:
        std::vector<T> g6(nV);

        // Initialize y0 from y
        for(int i = 0; i < nV; i++){
            y0[i] = y[i];
        }

        // First derivative
        this->Func(t, y0, parms, k0, x);
        for(int i = 0; i < nV; i++){
            k1[i] = k0[i];
        }

        // MAIN LOOP
        while(t < tEndOneStep - 1e-14 && nStep < nStepMax) {
            // Steps to compute k2..k7:
            for(int i = 0; i < nV; i++){
                yTemp[i] = y0[i] + h*(a21*k1[i]);
            }
            this->Func(t + c2*h, yTemp, parms, k2, x);

            for(int i = 0; i < nV; i++){
                yTemp[i] = y0[i] + h*(a31*k1[i] + a32*k2[i]);
            }
            this->Func(t + c3*h, yTemp, parms, k3, x);

            for(int i = 0; i < nV; i++){
                yTemp[i] = y0[i] + h*(a41*k1[i] + a42*k2[i] + a43*k3
