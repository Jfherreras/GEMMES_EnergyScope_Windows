#ifndef dopri_h
#define dopri_h

#include <iostream>
#include <fstream>
#include <math.h>
#include <cstring>
#include <stdio.h>
#include <string.h>
#include <vector>  // <-- ADDED for std::vector

#if UseRCPP==1
    #include "preprocRCPP_R.h"
#else
    #include "preproc.h"
#endif
#include "ODE.h"

// TO DO: ADD CONSTRUCTOR TO DEFINE ATOL, RTOL, NSTEPMAX, ETC
// TO DO: replace nVs by getNV()s
// TO DO: recycle k7 into k1 from one call to dopriOneStep to another instead of only within dopriOneStep
// TO DO: improve events management (especially eventVar)
// TO DO: make a solveLastPointNoEvents that only make one call to dopriOneIter for performance

using namespace std;

template<typename T>
class dopri : virtual public ODE<T> {
public:
    // ======================
    //  CONSTRUCTORS
    // ======================
    // default constructor
    dopri()
      : atol(1e-4), rtol(0),
        fac(0.85), facMin(0.1), facMax(3),
        nStepMax(1000),
        hInit(0.01), hMin(1e-4), hMax(0.1)
    {}

    // sets only h
    dopri(T hInitIn, T hMinIn, T hMaxIn)
      : atol(1e-4), rtol(0),
        fac(0.85), facMin(0.1), facMax(3),
        nStepMax(1000),
        hInit(hInitIn), hMin(hMinIn), hMax(hMaxIn)
    {}

    // sets all parameters
    dopri(T atolIn, T rtolIn, T facIn, T facMinIn, T facMaxIn,
          int nStepMaxIn, T hInitIn, T hMinIn, T hMaxIn)
      : atol(atolIn), rtol(rtolIn),
        fac(facIn), facMin(facMinIn), facMax(facMaxIn),
        nStepMax(nStepMaxIn),
        hInit(hInitIn), hMin(hMinIn), hMax(hMaxIn)
    {}

    // Bring members of ODE<T> into scope
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

    // ======================
    //  MAIN SOLVER
    // ======================
    void solve(const T* yInit, T* parms, T* out) override {
        // Gather dimensions
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        const int nRowOut = getNRowOut();
        T t = getTInit();

        // Replace VLA with std::vector
        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV), k4(nV), k5(nV), k6(nV), k7(nV);
        std::vector<T> x(nIV);
        std::vector<T> yTemp(nV), y0(nV), y4thOrder(nV);
        std::vector<T> tol(nV); // store tolerances

        T h = hInit;

        // Initialize first row of output with yInit
        for (int it = 0; it < nV; it++) {
            out[it] = yInit[it];
        }

        // Main Time-Stepping Loop
        for (int it = 1; it < nt; it++) {
            // Run eventTime & eventVar if needed
        #if UseEventTime
            makeEventTime(t, parms, &out[(it - 1) * nRowOut], x.data(), hInit);
        #endif
        #if UseEventVar
            makeEventVar(t, parms, &out[(it - 1) * nRowOut], x.data(), hInit);
        #endif

            // Step from t to t + getHOut() using dopri
            dopriOneStep(t,
                         &out[(it - 1) * nRowOut],
                         parms,
                         k0.data(), k1.data(), k2.data(), k3.data(),
                         k4.data(), k5.data(), k6.data(), k7.data(),
                         x.data(),
                         &out[it * nRowOut],
                         h,
                         tol.data(),
                         yTemp.data(),
                         y0.data(),
                         y4thOrder.data());

            // Complete output with ydot and x at time t
            completeOut(k0.data(), x.data(), &out[(it - 1) * nRowOut]);
        }

        // Complete out at the last point
        Func(t, &out[(nt - 1) * nRowOut], parms, k0.data(), x.data());
        completeOut(k0.data(), x.data(), &out[(nt - 1) * nRowOut]);
    }

    // ======================
    //  SOLVE ONLY LAST POINT
    // ======================
    void solveLastPoint(const T* yInit, T* parms, T* out) override {
        const int nV = getNV();
        const int nIV = getNIV();
        const int nt = getNt();
        T t = getTInit();

        // Replace VLA with std::vector
        std::vector<T> k0(nV), k1(nV), k2(nV), k3(nV), k4(nV), k5(nV), k6(nV), k7(nV);
        std::vector<T> x(nIV);
        std::vector<T> y(nV), yTemp(nV), y0(nV), y4thOrder(nV);
        std::vector<T> tol(nV);

        T h = hInit;

        // Initialize y from yInit
        for (int it = 0; it < nV; it++) {
            y[it] = yInit[it];
        }

        // Main Loop (we still do multiple steps if events are active)
        for (int step = 1; step < nt; step++) {
        #if UseEventTime
            makeEventTime(t, parms, y.data(), x.data(), hInit);
        #endif
        #if UseEventVar
            makeEventVar(t, parms, y.data(), x.data(), hInit);
        #endif
            // Step from t to t + getHOut()
            dopriOneStep(t,
                         y.data(),
                         parms,
                         k0.data(), k1.data(), k2.data(), k3.data(),
                         k4.data(), k5.data(), k6.data(), k7.data(),
                         x.data(),
                         y.data(), // read & write in same array
                         h,
                         tol.data(),
                         yTemp.data(),
                         y0.data(),
                         y4thOrder.data());
        }

        // Output final state to 'out'
        for (int it = 0; it < nV; it++) {
            out[it] = y[it];
        }

        // Fill derivatives / exog. vars
        Func(t, y.data(), parms, k0.data(), x.data());
        completeOut(k1.data(), x.data(), out);
    }

protected:
    // ======================
    //   SINGLE DOPRI STEP
    // ======================
    void dopriOneStep(T& t,
                      const T* y,
                      const T* parms,
                      T* k0, T* k1, T* k2, T* k3, T* k4, T* k5, T* k6, T* k7,
                      T* x,
                      T* out,
                      T& h,
                      T* tol,
                      T* yTemp,
                      T* y0,
                      T* y4thOrder)
    {
        const int nV = this->getNV();
        const int nIV = this->getNIV();
        T xTrashBuffer;
        // We'll store exog. variables in xTrash below. Use a local vector
        std::vector<T> xTrash(nIV);

        T tEndOneStep = t + this->getHOut();
        int nStep = 0;
        T error = 0.0;
        T alpha = 0.0;
        T hMem = 0.0;
        bool justRejected = false;
        bool truncatedH = false;

        // Initialize y0 from y
        for (int it = 0; it < nV; it++) {
            y0[it] = y[it];
        }

        // First derivative
        this->Func(t, y0, parms, k0, x);

        for (int it = 0; it < nV; it++) {
            k1[it] = k0[it];
        }

        // MAIN LOOP
        while (t < tEndOneStep - 1e-14 && nStep < nStepMax) {
            // Step 1
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a21 * k1[it]);
            }
            this->Func(t + c2 * h, yTemp, parms, k2, xTrash.data());

            // Step 2
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a31 * k1[it] + a32 * k2[it]);
            }
            this->Func(t + c3 * h, yTemp, parms, k3, xTrash.data());

            // Step 3
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a41 * k1[it] + a42 * k2[it] + a43 * k3[it]);
            }
            this->Func(t + c4 * h, yTemp, parms, k4, xTrash.data());

            // Step 4
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a51 * k1[it] + a52 * k2[it] + a53 * k3[it] + a54 * k4[it]);
            }
            this->Func(t + c5 * h, yTemp, parms, k5, xTrash.data());

            // Step 5
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a61 * k1[it] + a62 * k2[it] + a63 * k3[it] + a64 * k4[it] + a65 * k5[it]);
            }
            this->Func(t + c6 * h, yTemp, parms, k6, xTrash.data());

            // Step 6
            for (int it = 0; it < nV; it++) {
                yTemp[it] = y0[it] + h * (a71 * k1[it] + a73 * k3[it] + a74 * k4[it] + a75 * k5[it] + a76 * k6[it]);
            }
            this->Func(t + c7 * h, yTemp, parms, k7, xTrash.data());

            // 4th-order solution for error estimation
            for (int it = 0; it < nV; it++) {
                y4thOrder[it] = y0[it] + h * (b1p * k1[it] + b3p * k3[it] + b4p * k4[it]
                                             + b5p * k5[it] + b6p * k6[it] + b7p * k7[it]);
            }

            // Evaluate error
            error = 0.0;
            for (int it = 0; it < nV; it++) {
                tol[it] = atol + rtol * std::max(std::abs(yTemp[it]), std::abs(y4thOrder[it]));
            }
            for (int it = 0; it < nV; it++) {
                T diff = (yTemp[it] - y4thOrder[it]) / tol[it];
                error += diff * diff;
            }
            error = std::sqrt(error / nV);

            // Step acceptance / rejection
            alpha = fac * std::pow((1 / error), 1.0 / 5.0);
            if (error <= 1) {
                // Accept step
                justRejected = false;
                t += h; // move forward
                for (int it = 0; it < nV; it++) {
                    k1[it] = k7[it];  // recycle last derivative
                    y0[it] = yTemp[it];
                }
            } else {
                // Reject step
                justRejected = true;
            }

            // Adjust step size
            if (alpha < facMin || (alpha != alpha)) {
                h *= facMin;
            } else if (justRejected && alpha > 1.0) {
                // do not increase h right after a rejection
            } else if (alpha > facMax) {
                h *= facMax;
            } else {
                h *= alpha;
            }

            if (h > hMax) {
                h = hMax;
            } else if (h < hMin) {
                h = hMin;
            }

            // If we’re about to overshoot tEndOneStep, shorten h
            if ((t + h) > tEndOneStep && (t < tEndOneStep)) {
                hMem = h;
                h = tEndOneStep - t;
                truncatedH = true;
            }

            nStep++;
        }

        if (truncatedH) {
            h = hMem;  // restore the old step for next iteration
        }

        // Final result of this sub-step
        for (int it = 0; it < nV; it++) {
            out[it] = yTemp[it];
        }

        // If we exceeded max steps, throw
        if (nStep >= nStepMax - 1) {
            throw std::runtime_error(
                "Maximum number of steps reached. Possibly due to stiffness. "
                "Reduce time step, increase maximum steps, or use a stiff solver."
            );
        }
    }

    // ======================
    //  DATA MEMBERS
    // ======================
    T atol;
    T rtol;
    T fac;
    T facMin;
    T facMax;
    T nStepMax;
    T hInit;
    T hMin;
    T hMax;

    // Coefficients for RK
    // Terms set to zero are included for clarity
    const T a21 = (1.0/5.0);
    const T a31 = (3.0/40.0);
    const T a32 = (9.0/40.0);
    const T a41 = (44.0/45.0);
    const T a42 = (-56.0/15.0);
    const T a43 = (32.0/9.0);
    const T a51 = (19372.0/6561.0);
    const T a52 = (-25360.0/2187.0);
    const T a53 = (64448.0/6561.0);
    const T a54 = (-212.0/729.0);
    const T a61 = (9017.0/3168.0);
    const T a62 = (-355.0/33.0);
    const T a63 = (46732.0/5247.0);
    const T a64 = (49.0/176.0);
    const T a65 = (-5103.0/18656.0);
    const T a71 = (35.0/384.0);
    const T a72 = (0.0);
    const T a73 = (500.0/1113.0);
    const T a74 = (125.0/192.0);
    const T a75 = (-2187.0/6784.0);
    const T a76 = (11.0/84.0);

    const T c2  = (1.0/5.0);
    const T c3  = (3.0/10.0);
    const T c4  = (4.0/5.0);
    const T c5  = (8.0/9.0);
    const T c6  = (1.0);
    const T c7  = (1.0);

    // b1...b7 not used in practice, included for clarity
    const T b1  = (35.0/384.0);
    const T b2  = (0.0);
    const T b3  = (500.0/1113.0);
    const T b4  = (125.0/192.0);
    const T b5  = (-2187.0/6784.0);
    const T b6  = (11.0/84.0);
    const T b7  = (0.0);

    const T b1p = (5179.0/57600.0);
    const T b2p = (0.0);
    const T b3p = (7571.0/16695.0);
    const T b4p = (393.0/640.0);
    const T b5p = (-92097.0/339200.0);
    const T b6p = (187.0/2100.0);
    const T b7p = (1.0/40.0);
};

#endif
