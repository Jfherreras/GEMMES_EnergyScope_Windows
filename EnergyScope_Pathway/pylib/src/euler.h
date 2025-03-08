#ifndef EULER_h
#define EULER_h

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <vector>

#if UseRCPP==1
  #include "preprocRCPP_R.h"
#else
  #include "preproc.h"
#endif
#include "ODE.h"

using namespace std;

template<typename T>
class Euler : virtual public ODE<T> {
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
    // Main solver
    void solve(const T* yInit, T* parms, T* out) override {
        double t = getTInit();
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();

        // Replace variable-length arrays with vectors
        std::vector<T> yDot(nV);
        std::vector<T> x(nIV);

        // Initialize the first row
        for(int i = 0; i < nV; i++){
            out[i] = yInit[i];
        }

        // Main loop
        for(int it = 1; it < nt; it++){
        #if UseEventTime
            makeEventTime(t, parms, &out[(it - 1)*getNRowOut()], x.data(), getHOut());
        #endif
        #if UseEventVar
            makeEventVar(t, parms, &out[(it - 1)*getNRowOut()], x.data(), getHOut());
        #endif

            // Euler step
            EulerOneStep(
                t,
                &out[(it - 1) * getNRowOut()],
                parms,
                yDot.data(),
                x.data(),
                &out[it * getNRowOut()]
            );

            // complete output
            completeOut(yDot.data(), x.data(), &out[(it - 1)*getNRowOut()]);

            t += getHOut();
        }

        // final
        Func(t, &out[(nt - 1)*getNRowOut()], parms, yDot.data(), x.data());
        completeOut(yDot.data(), x.data(), &out[(nt - 1)*getNRowOut()]);
    }

    // Same but returns only last point
    void solveLastPoint(const T* yInit, T* parms, T* out) override {
        double t = getTInit();
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();

        std::vector<T> yDot(nV);
        std::vector<T> x(nIV);
        std::vector<T> y(nV);

        for(int i = 0; i < nV; i++){
            y[i] = yInit[i];
        }

        for(int step = 1; step < nt; step++){
        #if UseEventTime
            makeEventTime(t, parms, y.data(), x.data(), getHOut());
        #endif
        #if UseEventVar
            makeEventVar(t, parms, y.data(), x.data(), getHOut());
        #endif

            // Euler step in place (y->y)
            EulerOneStep(
                t,
                y.data(),
                parms,
                yDot.data(),
                x.data(),
                y.data() // overwriting itself
            );

            t += getHOut();
        }

        for(int i = 0; i < nV; i++){
            out[i] = y[i];
        }

        Func(t, y.data(), parms, yDot.data(), x.data());
        completeOut(yDot.data(), x.data(), out);
    }

protected:
    // The actual Euler step function
    void EulerOneStep(
        const T t,
        const T* yIn,
        const T* parms,
        T* yDot,
        T* x,
        T* yOut
    ){
        // same logic as your code
        // e.g.:
        // Func(t, yIn, parms, yDot, x);
        // for (int i = 0; i < getNV(); i++) {
        //     yOut[i] = yIn[i] + getHOut() * yDot[i];
        // }
    }
};

#endif
