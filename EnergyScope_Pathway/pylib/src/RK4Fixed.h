#ifndef RK4Fixed_h
#define RK4Fixed_h

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
    void solve(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        const int nRowOut = getNRowOut();
        T t = getTInit();

        // Switch to vectors
        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV);
        std::vector<T> x(nIV);

        // init
        for(int i = 0; i < nV; i++){
            out[i] = yInit[i];
        }

        // main loop
        for(int it = 1; it < nt; it++){
        #if UseEventTime
            makeEventTime(t, parms, &out[(it - 1) * nRowOut], x.data(), getHOut());
        #endif
        #if UseEventVar
            makeEventVar(t, parms, &out[(it - 1) * nRowOut], x.data(), getHOut());
        #endif

            RK4OneStep(
                t,
                &out[(it - 1)*nRowOut],
                parms,
                k0.data(), k1.data(), k2.data(), k3.data(),
                x.data(),
                &out[it*nRowOut]
            );

            completeOut(k0.data(), x.data(), &out[(it - 1)*nRowOut]);
            t += getHOut();
        }

        Func(t, &out[(nt - 1) * nRowOut], parms, k0.data(), x.data());
        completeOut(k0.data(), x.data(), &out[(nt - 1) * nRowOut]);
    }

    void solveLastPoint(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        T t = getTInit();

        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV);
        std::vector<T> x(nIV);
        std::vector<T> yIn(nV), yOut(nV);

        for(int i = 0; i < nV; i++){
            yIn[i] = yInit[i];
        }

        for(int step = 1; step < nt; step++){
        #if UseEventTime
            makeEventTime(t, parms, yIn.data(), x.data(), getHOut());
        #endif
        #if UseEventVar
            makeEventVar(t, parms, yIn.data(), x.data(), getHOut());
        #endif

            RK4OneStep(
                t,
                yIn.data(),
                parms,
                k0.data(), k1.data(), k2.data(), k3.data(),
                x.data(),
                yOut.data()
            );
            for(int i=0; i<nV; i++){
                yIn[i] = yOut[i];
            }
            t += getHOut();
        }

        for(int i=0; i<nV; i++){
            out[i] = yIn[i];
        }

        Func(t, yIn.data(), parms, k0.data(), x.data());
        completeOut(k0.data(), x.data(), out);
    }

protected:
    void RK4OneStep(
        T t,
        const T* yIn,
        const T* parms,
        T* k0, T* k1, T* k2, T* k3,
        T* x,
        T* yOut
    ){
        // standard RK4 step
        // e.g.:
        // Func(t, yIn, parms, k0, x);
        // for(int i=0; i<getNV(); i++){
        //     ...
        // }
    }
};

#endif
