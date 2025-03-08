#ifndef RK4Fixed_h
#define RK4Fixed_h

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <vector>  // <-- added for std::vector usage

#if UseRCPP==1
    #include "preprocRCPP_R.h"
#else
    #include "preproc.h"
#endif
#include "ODE.h"

using namespace std;

template<typename T>
class RK4Fixed : virtual public ODE<T> {
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

public:
    // ================================
    // MAIN SOLVER (entire trajectory)
    // ================================
    void solve(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        const int nRowOut = getNRowOut();
        T t = getTInit();

        // Replace arrays with std::vector
        std::vector<T> k1(nV), k2(nV), k3(nV), k4(nV);
        std::vector<T> x(nIV);

        // Initialize first output row
        for (int it = 0; it < nV; it++) {
            out[it] = yInit[it];
        }

        // Main time-stepping loop
        for (int it = 1; it < nt; it++) {
            // Run events if needed
        #if UseEventTime
            makeEventTime(t, parms, &out[(it - 1) * nRowOut], x.data(), getHOut());
        #endif
        #if UseEventVar
            makeEventVar(t, parms, &out[(it - 1) * nRowOut], x.data(), getHOut());
        #endif

            // Perform one RK4 step
            RK4OneStep(t,
                       &out[(it - 1) * nRowOut],
                       parms,
                       k1.data(), k2.data(), k3.data(), k4.data(),
                       x.data(),
                       &out[it * nRowOut]);

            // Fill out ydot, x at time t
            completeOut(k1.data(), x.data(), &out[(it - 1) * nRowOut]);

            // Increment time
            t += getHOut();
        }

        // Complete out at the last point
        Func(t, &out[(nt - 1) * nRowOut], parms, k1.data(), x.data());
        completeOut(k1.data(), x.data(), &out[(nt - 1) * nRowOut]);
    }

    // =========================================
    // SOLVE ONLY THE LAST POINT OF TRAJECTORY
    // =========================================
    void solveLastPoint(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        T t = getTInit();

        // Replace arrays with std::vector
        std::vector<T> k1(nV), k2(nV), k3(nV), k4(nV);
        std::vector<T> x(nIV);
        std::vector<T> yIn(nV), yOut(nV);

        // Initialize from yInit
        for (int it = 0; it < nV; it++) {
            yIn[it] = yInit[it];
        }

        // Main loop
        for (int step = 1; step < nt; step++) {
        #if UseEventTime
            makeEventTime(t, parms, yIn.data(), x.data(), getHOut());
        #endif
        #if
