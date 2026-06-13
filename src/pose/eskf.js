/**
 * Error-State Kalman Filter — forward pass for body-pose estimation.
 *
 * Estimates the drone's position, velocity, and attitude from IMU propagation
 * corrected by GPS, baro, and attitude-prior measurements.
 *
 * The filter tracks a nominal state x = {p, v, q} and a 9-dimensional error
 * state δx = [δp(3), δv(3), δθ(3)] with covariance P(9×9). After each update
 * the error is injected into the nominal state and reset to zero.
 *
 * Conventions:
 *   World: NED    Body: FRD
 *   Quaternions: Hamilton, body(FRD)→world(NED), scalar-first [w,x,y,z]
 *   Gravity: [0, 0, +9.80665] m/s² in NED
 */

import { quatToRot, quatMultiply, quatFromAxisAngle, strapdownPropagate } from "./imuMechanization.js";

// ---------------------------------------------------------------------------
// Matrix helpers
// ---------------------------------------------------------------------------

function matIdentity(n) {
    const I = new Array(n);
    for (let i = 0; i < n; i++) {
        I[i] = new Array(n).fill(0);
        I[i][i] = 1;
    }
    return I;
}

/** C = A × B, all square n×n */
function matMul(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n).fill(0);
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            if (aik === 0) continue;
            for (let j = 0; j < n; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}

/** C = A + B, same dimensions */
function matAdd(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n);
        for (let j = 0; j < n; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

/** A^T */
function matTranspose(A) {
    const n = A.length;
    const T = new Array(n);
    for (let i = 0; i < n; i++) {
        T[i] = new Array(n);
        for (let j = 0; j < n; j++) {
            T[i][j] = A[j][i];
        }
    }
    return T;
}

/** Invert a symmetric positive-definite matrix via Gauss-Jordan (n ≤ ~30). */
function matInvertSym(A) {
    const n = A.length;
    const aug = new Array(n);
    for (let i = 0; i < n; i++) {
        aug[i] = new Array(2 * n);
        for (let j = 0; j < n; j++) {
            aug[i][j] = A[i][j];
            aug[i][j + n] = i === j ? 1 : 0;
        }
    }

    for (let i = 0; i < n; i++) {
        let maxRow = i;
        let maxVal = Math.abs(aug[i][i]);
        for (let r = i + 1; r < n; r++) {
            if (Math.abs(aug[r][i]) > maxVal) {
                maxVal = Math.abs(aug[r][i]);
                maxRow = r;
            }
        }
        if (maxRow !== i) {
            const tmp = aug[i];
            aug[i] = aug[maxRow];
            aug[maxRow] = tmp;
        }

        const pivot = aug[i][i];
        if (Math.abs(pivot) < 1e-30) continue;

        for (let j = 0; j < 2 * n; j++) aug[i][j] /= pivot;

        for (let r = 0; r < n; r++) {
            if (r === i) continue;
            const factor = aug[r][i];
            for (let j = 0; j < 2 * n; j++) aug[r][j] -= factor * aug[i][j];
        }
    }

    const inv = new Array(n);
    for (let i = 0; i < n; i++) {
        inv[i] = new Array(n);
        for (let j = 0; j < n; j++) inv[i][j] = aug[i][j + n];
    }
    return inv;
}

// ---------------------------------------------------------------------------
// Error-state transition matrix F (9×9)
// ---------------------------------------------------------------------------

/**
 * Build the error-state transition matrix F_k for one IMU step.
 *
 * Continuous dynamics (simplified, no bias states):
 *   δṗ = δv
 *   δv̇ = −R·skew(ã)·δθ + n_a
 *   δθ̇ = −skew(ω̃)·δθ + n_g
 *
 * Discretised with explicit Euler:
 *   F = I + F_c·dt
 *
 * @param {number[]} q   nominal quaternion [w,x,y,z] at step start
 * @param {number[]} accel  de-biased accel in body FRD [ax,ay,az] m/s²
 * @param {number[]} omega  de-biased gyro in body FRD [wx,wy,wz] rad/s
 * @param {number} dt  time step in seconds
 * @returns {number[][]} 9×9 F matrix
 */
function buildTransition(q, accel, omega, dt) {
    const F = matIdentity(9);
    const R = quatToRot(q);

    const wx = omega[0], wy = omega[1], wz = omega[2];
    const ax = accel[0], ay = accel[1], az = accel[2];

    const dt2h = 0.5 * dt * dt;

    // δp ← δv·dt
    F[0][3] = dt;
    F[1][4] = dt;
    F[2][5] = dt;

    // δp ← δθ coupling: −½·R·skew(ã)·dt²
    // skew(ã) = [[0,−az,ay],[az,0,−ax],[−ay,ax,0]]
    // R·skew(ã) = ... let me compute this carefully
    const s00 = 0,           s01 = -az,        s02 = ay;
    const s10 = az,          s11 = 0,          s12 = -ax;
    const s20 = -ay,         s21 = ax,         s22 = 0;

    // R · skew(ã)
    const rs00 = R[0][0] * s00 + R[0][1] * s10 + R[0][2] * s20;
    const rs01 = R[0][0] * s01 + R[0][1] * s11 + R[0][2] * s21;
    const rs02 = R[0][0] * s02 + R[0][1] * s12 + R[0][2] * s22;
    const rs10 = R[1][0] * s00 + R[1][1] * s10 + R[1][2] * s20;
    const rs11 = R[1][0] * s01 + R[1][1] * s11 + R[1][2] * s21;
    const rs12 = R[1][0] * s02 + R[1][1] * s12 + R[1][2] * s22;
    const rs20 = R[2][0] * s00 + R[2][1] * s10 + R[2][2] * s20;
    const rs21 = R[2][0] * s01 + R[2][1] * s11 + R[2][2] * s21;
    const rs22 = R[2][0] * s02 + R[2][1] * s12 + R[2][2] * s22;

    F[0][6] = -rs00 * dt2h;  F[0][7] = -rs01 * dt2h;  F[0][8] = -rs02 * dt2h;
    F[1][6] = -rs10 * dt2h;  F[1][7] = -rs11 * dt2h;  F[1][8] = -rs12 * dt2h;
    F[2][6] = -rs20 * dt2h;  F[2][7] = -rs21 * dt2h;  F[2][8] = -rs22 * dt2h;

    // δv ← δθ coupling: −R·skew(ã)·dt
    F[3][6] = -rs00 * dt;  F[3][7] = -rs01 * dt;  F[3][8] = -rs02 * dt;
    F[4][6] = -rs10 * dt;  F[4][7] = -rs11 * dt;  F[4][8] = -rs12 * dt;
    F[5][6] = -rs20 * dt;  F[5][7] = -rs21 * dt;  F[5][8] = -rs22 * dt;

    // δθ ← −skew(ω̃)·δθ·dt: F[6..8][6..8] = I − skew(omega)·dt
    F[6][6] = 1;      F[6][7] = wz * dt;  F[6][8] = -wy * dt;
    F[7][6] = -wz * dt;  F[7][7] = 1;      F[7][8] = wx * dt;
    F[8][6] = wy * dt;  F[8][7] = -wx * dt;  F[8][8] = 1;

    return F;
}

/**
 * Build the process noise covariance Q_k (9×9) for one IMU step.
 *
 * @param {number[][]} R_wb  3×3 rotation matrix body→world
 * @param {number} sigmaAcc  accel noise (m/s²)
 * @param {number} sigmaGyro gyro noise (rad/s)
 * @param {number} dt  time step (s)
 * @returns {number[][]} 9×9 Q matrix
 */
function buildProcessNoise(R_wb, sigmaAcc, sigmaGyro, dt) {
    const Q = new Array(9);
    for (let i = 0; i < 9; i++) Q[i] = new Array(9).fill(0);

    const sa2 = sigmaAcc * sigmaAcc;
    const sg2 = sigmaGyro * sigmaGyro;
    const dt2 = dt * dt;
    const dt3 = dt2 * dt / 3;
    const dt4 = dt2 * dt2 / 4;

    // Position noise from accel: σa²·dt⁴/4 · I3 (in world frame)
    const pNoise = sa2 * dt4;
    Q[0][0] = pNoise;  Q[1][1] = pNoise;  Q[2][2] = pNoise;

    // Velocity noise from accel: σa²·dt² · I3
    const vNoise = sa2 * dt2;
    Q[3][3] = vNoise;  Q[4][4] = vNoise;  Q[5][5] = vNoise;

    // Cross terms p-v: σa²·dt³/2
    const pvNoise = sa2 * dt3 / 2;
    Q[0][3] = Q[3][0] = pvNoise;
    Q[1][4] = Q[4][1] = pvNoise;
    Q[2][5] = Q[5][2] = pvNoise;

    // Attitude noise from gyro: σg²·dt² · I3
    const gNoise = sg2 * dt2;
    Q[6][6] = gNoise;  Q[7][7] = gNoise;  Q[8][8] = gNoise;

    return Q;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Initialize an ESKF instance.
 *
 * @param {object} opts
 * @param {number[]}  opts.p0      initial position NED [n,e,d] (m)
 * @param {number[]}  opts.v0      initial velocity NED [vn,ve,vd] (m/s)
 * @param {number[]}  opts.q0      initial attitude quaternion [w,x,y,z]
 * @param {number}    [opts.sigmaPos=5]    initial position uncertainty (m)
 * @param {number}    [opts.sigmaVel=2]    initial velocity uncertainty (m/s)
 * @param {number}    [opts.sigmaAtt=0.2]  initial attitude uncertainty (rad)
 * @returns {object} ESKF instance
 */
export function createEskf({ p0, v0, q0, sigmaPos = 5, sigmaVel = 2, sigmaAtt = 0.2 }) {
    const P = matIdentity(9);
    P[0][0] = P[1][1] = P[2][2] = sigmaPos * sigmaPos;
    P[3][3] = P[4][4] = P[5][5] = sigmaVel * sigmaVel;
    P[6][6] = P[7][7] = P[8][8] = sigmaAtt * sigmaAtt;

    return {
        /** Nominal state */
        p: p0.slice(),
        v: v0.slice(),
        q: q0.slice(),

        /** Error-state covariance (9×9) */
        P,

        /** Runtime options */
        sigmaAcc: 0.35,
        sigmaGyro: 0.015,
    };
}

/**
 * Predict step: propagate nominal state via IMU strapdown and advance covariance.
 *
 * @param {object} eskf
 * @param {number[]} omega  gyro reading [wx,wy,wz] body FRD rad/s
 * @param {number[]} accel  accel reading [ax,ay,az] body FRD m/s²
 * @param {number}   dt     time step in seconds
 */
export function eskfPredict(eskf, omega, accel, dt) {
    const { q, v, p, sigmaAcc, sigmaGyro } = eskf;

    // De-biased IMU (Phase 1: no bias estimates, assume pre-arm calibrated)
    const omegaD = omega;
    const accelD = accel;

    // Specific force = negative of de-biased sensor reading (MEMS convention)
    const sfX = -accelD[0];
    const sfY = -accelD[1];
    const sfZ = -accelD[2];

    // Build F and Q using specific force (the true body-frame acceleration minus gravity)
    const F = buildTransition(q, [sfX, sfY, sfZ], omegaD, dt);
    const R_wb = quatToRot(q);
    const Q = buildProcessNoise(R_wb, sigmaAcc, sigmaGyro, dt);

    // Propagate nominal state via strapdown
    const next = strapdownPropagate(omegaD, accelD, q, v, p, dt);
    eskf.p = next.p;
    eskf.v = next.v;
    eskf.q = next.q;

    // Propagate covariance: P = F·P·Fᵀ + Q
    const FP = matMul(F, eskf.P);
    const FPFt = matMul(FP, matTranspose(F));
    eskf.P = matAdd(FPFt, Q);

    // Covariance symmetry forcing
    for (let i = 0; i < 9; i++) {
        for (let j = i + 1; j < 9; j++) {
            const avg = (eskf.P[i][j] + eskf.P[j][i]) * 0.5;
            eskf.P[i][j] = avg;
            eskf.P[j][i] = avg;
        }
    }
}

/**
 * Update step: apply a measurement factor.
 *
 * Each factor provides { H, R, residual(z, x) }.
 *
 * @param {object}   eskf
 * @param {object}   factor  from measurements.js factory functions
 * @param {number[]} z       measurement vector
 * @param {number}   [gate]  chi-square gate factor (default 3.0). Measurement rejected if testRatio > 1.
 * @returns {boolean} true if measurement was accepted
 */
export function eskfUpdate(eskf, factor, z, gate = 3.0) {
    const x = { p: eskf.p, v: eskf.v, q: eskf.q };

    // Innovation r = z − h(x)
    const r = factor.residual(z, x);
    const H = factor.H;   // array of rows, each row is a 9-element array
    const R = factor.R;
    const m = r.length;   // measurement dimension

    // S = H·P·Hᵀ + R
    const PHt = new Array(9);
    for (let i = 0; i < 9; i++) {
        PHt[i] = new Array(m).fill(0);
        for (let k = 0; k < 9; k++) {
            const pik = eskf.P[i][k];
            if (pik === 0) continue;
            for (let j = 0; j < m; j++) {
                PHt[i][j] += pik * H[j][k];
            }
        }
    }

    const S = new Array(m);
    for (let i = 0; i < m; i++) {
        S[i] = new Array(m).fill(0);
        for (let j = 0; j < m; j++) {
            for (let k = 0; k < 9; k++) {
                S[i][j] += H[i][k] * PHt[k][j];
            }
            S[i][j] += R[i][j];
        }
    }

    // Chi-square gate
    const S_inv = matInvertSym(S);
    let mahal = 0;
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
            mahal += r[i] * S_inv[i][j] * r[j];
        }
    }
    const gateSq = (gate * gate) * m;  // m degrees of freedom
    if (mahal > gateSq) return false;

    // Kalman gain: K = P·Hᵀ·S⁻¹
    const K = new Array(9);
    for (let i = 0; i < 9; i++) {
        K[i] = new Array(m);
        for (let j = 0; j < m; j++) {
            let s = 0;
            for (let k = 0; k < m; k++) {
                s += PHt[i][k] * S_inv[k][j];
            }
            K[i][j] = s;
        }
    }

    // Error-state correction: δx = K·r
    const dx = new Array(9).fill(0);
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < m; j++) {
            dx[i] += K[i][j] * r[j];
        }
    }

    // Inject error into nominal state
    eskf.p[0] += dx[0];  eskf.p[1] += dx[1];  eskf.p[2] += dx[2];
    eskf.v[0] += dx[3];  eskf.v[1] += dx[4];  eskf.v[2] += dx[5];

    // Attitude injection: q ← exp(½·δθ) ⊗ q
    const dtheta = [dx[6], dx[7], dx[8]];
    const dthetaNorm = Math.sqrt(dtheta[0] ** 2 + dtheta[1] ** 2 + dtheta[2] ** 2);
    if (dthetaNorm > 1e-12) {
        const axis = [dtheta[0] / dthetaNorm, dtheta[1] / dthetaNorm, dtheta[2] / dthetaNorm];
        const dq = quatFromAxisAngle(axis, dthetaNorm);
        const newQ = quatMultiply(dq, eskf.q);
        const nq = Math.sqrt(newQ[0] ** 2 + newQ[1] ** 2 + newQ[2] ** 2 + newQ[3] ** 2);
        eskf.q = [newQ[0] / nq, newQ[1] / nq, newQ[2] / nq, newQ[3] / nq];
    }

    // Joseph-form covariance update: P = (I−KH)·P·(I−KH)ᵀ + K·R·Kᵀ
    const I_KH = matIdentity(9);
    for (let i = 0; i < 9; i++) {
        for (let j = 0; j < 9; j++) {
            for (let k = 0; k < m; k++) {
                I_KH[i][j] -= K[i][k] * H[k][j];
            }
        }
    }

    const IKH_P = matMul(I_KH, eskf.P);
    const IKH_P_IKHt = matMul(IKH_P, matTranspose(I_KH));

    const KRKt = new Array(9);
    for (let i = 0; i < 9; i++) {
        KRKt[i] = new Array(9).fill(0);
        for (let j = 0; j < 9; j++) {
            for (let ki = 0; ki < m; ki++) {
                for (let kj = 0; kj < m; kj++) {
                    KRKt[i][j] += K[i][ki] * R[ki][kj] * K[j][kj];
                }
            }
        }
    }

    eskf.P = matAdd(IKH_P_IKHt, KRKt);

    // Symmetry forcing
    for (let i = 0; i < 9; i++) {
        for (let j = i + 1; j < 9; j++) {
            const avg = (eskf.P[i][j] + eskf.P[j][i]) * 0.5;
            eskf.P[i][j] = avg;
            eskf.P[j][i] = avg;
        }
    }

    // Variance floor
    for (let i = 0; i < 9; i++) {
        if (eskf.P[i][i] < 1e-6) eskf.P[i][i] = 1e-6;
    }

    return true;
}

/**
 * Extract current nominal state.
 *
 * @param {object} eskf
 * @returns {{ p: number[], v: number[], q: number[], sigmaPos: number, sigmaAtt: number }}
 */
export function eskfGetState(eskf) {
    return {
        p: eskf.p.slice(),
        v: eskf.v.slice(),
        q: eskf.q.slice(),
        sigmaPos: Math.sqrt(
            Math.max(0, (eskf.P[0][0] + eskf.P[1][1] + eskf.P[2][2]) / 3),
        ),
        sigmaAtt: Math.sqrt(
            Math.max(0, (eskf.P[6][6] + eskf.P[7][7] + eskf.P[8][8]) / 3),
        ) * (180 / Math.PI),
    };
}
