/**
 * IMU on-manifold preintegration between keyframes.
 * Implements Forster et al. (TRO 2017) mid-point integration.
 *
 * Accumulates IMU measurements between two keyframes (e.g. GPS fixes
 * ~100 ms apart, ~50 IMU samples) into a compact delta {ΔR, Δv, Δp}
 * with a 9×9 covariance, decoupled from the linearisation point.
 *
 * Conventions:
 *   Body frame:  FRD (Forward=X, Right=Y, Down=Z)
 *   World frame: NED (North, East, Down)
 *   Quaternion:  Hamilton, body(FRD) → world(NED), scalar-first [w, x, y, z]
 *   Gravity:     [0, 0, +9.80665] in NED
 *
 * @module imuPreintegration
 */

import { skew, rotToQuat } from './imuMechanization.js';

// ---------------------------------------------------------------------------
// Default noise densities
// ---------------------------------------------------------------------------

/** Accelerometer noise density [m/s² / √Hz] */
const SIGMA_ACC = 0.01;

/** Gyroscope noise density [rad/s / √Hz] */
const SIGMA_GYRO = 0.001;

/** Gravitational acceleration in world NED [m/s²] */
const G_WORLD = [0, 0, 9.80665];

// ---------------------------------------------------------------------------
// 3×3 matrix helpers
// ---------------------------------------------------------------------------

/** @returns {number[][]} 3×3 identity */
function mat3Identity() {
    return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
}

/** Deep-copy a 3×3 matrix */
function mat3Copy(A) {
    return [A[0].slice(), A[1].slice(), A[2].slice()];
}

/** C = s × A  (scalar × 3×3 matrix) */
function mat3Scale(A, s) {
    return [
        [A[0][0] * s, A[0][1] * s, A[0][2] * s],
        [A[1][0] * s, A[1][1] * s, A[1][2] * s],
        [A[2][0] * s, A[2][1] * s, A[2][2] * s],
    ];
}

/** C = A × B  (3×3 matrix multiplication) */
function mat3Mul(A, B) {
    return [
        [
            A[0][0] * B[0][0] + A[0][1] * B[1][0] + A[0][2] * B[2][0],
            A[0][0] * B[0][1] + A[0][1] * B[1][1] + A[0][2] * B[2][1],
            A[0][0] * B[0][2] + A[0][1] * B[1][2] + A[0][2] * B[2][2],
        ],
        [
            A[1][0] * B[0][0] + A[1][1] * B[1][0] + A[1][2] * B[2][0],
            A[1][0] * B[0][1] + A[1][1] * B[1][1] + A[1][2] * B[2][1],
            A[1][0] * B[0][2] + A[1][1] * B[1][2] + A[1][2] * B[2][2],
        ],
        [
            A[2][0] * B[0][0] + A[2][1] * B[1][0] + A[2][2] * B[2][0],
            A[2][0] * B[0][1] + A[2][1] * B[1][1] + A[2][2] * B[2][1],
            A[2][0] * B[0][2] + A[2][1] * B[1][2] + A[2][2] * B[2][2],
        ],
    ];
}

/** A^T  (3×3 transpose) */
function mat3Transpose(A) {
    return [
        [A[0][0], A[1][0], A[2][0]],
        [A[0][1], A[1][1], A[2][1]],
        [A[0][2], A[1][2], A[2][2]],
    ];
}

/** A · v  (3×3 matrix × 3-vector) */
function mat3VecMul(A, v) {
    return [
        A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
        A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
        A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
    ];
}

// ---------------------------------------------------------------------------
// Rodrigues formula — SO(3) exponential map
// ---------------------------------------------------------------------------

/**
 * exp(φ) → 3×3 rotation matrix.
 * For θ → 0 returns I + skew(φ).
 *
 * @param {number[]} phi  Rotation vector [φx, φy, φz] in radians
 * @returns {number[][]}  3×3 rotation matrix
 */
function expRodrigues(phi) {
    const [x, y, z] = phi;
    const t2 = x * x + y * y + z * z;

    if (t2 < 1e-14) {
        return [
            [1, -z, y],
            [z, 1, -x],
            [-y, x, 1],
        ];
    }

    const theta = Math.sqrt(t2);
    const s = Math.sin(theta);
    const c = Math.cos(theta);
    const a = s / theta;
    const b = (1 - c) / t2;

    const xy = x * y, xz = x * z, yz = y * z;

    return [
        [1 - b * (y * y + z * z), -a * z + b * xy, a * y + b * xz],
        [a * z + b * xy, 1 - b * (x * x + z * z), -a * x + b * yz],
        [-a * y + b * xz, a * x + b * yz, 1 - b * (x * x + y * y)],
    ];
}

// ---------------------------------------------------------------------------
// Simplified right Jacobian — Jr(φ) ≈ I − ½·skew(φ)
// ---------------------------------------------------------------------------

/**
 * First-order approximation of the SO(3) right Jacobian.
 * Accurately sufficient for small per-step rotation vectors (|φ| ≪ 1).
 *
 * @param {number[]} phi  Rotation vector [φx, φy, φz] in radians
 * @returns {number[][]}  3×3 matrix
 */
function jrSimplified(phi) {
    const [x, y, z] = phi;
    return [
        [1, 0.5 * z, -0.5 * y],
        [-0.5 * z, 1, 0.5 * x],
        [0.5 * y, -0.5 * x, 1],
    ];
}

// ---------------------------------------------------------------------------
// Arbitrary-size matrix helpers (for 9×9 and 9×6)
// ---------------------------------------------------------------------------

/** @returns {number[][]} n×n zero matrix */
function matZero(n) {
    const Z = new Array(n);
    for (let i = 0; i < n; i++) Z[i] = new Array(n).fill(0);
    return Z;
}

/** @returns {number[][]} n×n identity */
function matIdentity(n) {
    const I = new Array(n);
    for (let i = 0; i < n; i++) {
        I[i] = new Array(n).fill(0);
        I[i][i] = 1;
    }
    return I;
}

/** C = A + B  (both n×n) */
function matAdd(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n);
        for (let j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
    }
    return C;
}

/**
 * C = A · B · A^T  (both n×n).
 * Computed as (A·B) · A^T to save one full multiply over A·(B·A^T).
 *
 * @param {number[][]} A
 * @param {number[][]} B
 * @returns {number[][]}
 */
function matMulABAT(A, B) {
    const n = A.length;

    const temp = matZero(n);
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            if (aik === 0) continue;
            for (let j = 0; j < n; j++) temp[i][j] += aik * B[k][j];
        }
    }

    const result = matZero(n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) sum += temp[i][k] * A[j][k];
            result[i][j] = sum;
        }
    }
    return result;
}

/**
 * B · diag(d) · B^T  where B is n×m and d is an m-element array.
 *
 * @param {number[][]} B  n×m matrix
 * @param {number[]}   d  diagonal entries [m]
 * @returns {number[][]}  n×n result
 */
function matMulDiagBT(B, d) {
    const n = B.length;
    const m = d.length;
    const result = matZero(n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < m; k++) sum += B[i][k] * d[k] * B[j][k];
            result[i][j] = sum;
        }
    }
    return result;
}

/**
 * Copy a srcRows×srcCols sub-matrix into a larger destination matrix (in-place).
 *
 * @param {number[][]} dst  Destination matrix
 * @param {number}     r0   Starting row in dst
 * @param {number}     c0   Starting column in dst
 * @param {number[][]} src  Source matrix
 */
function matSetBlock(dst, r0, c0, src) {
    const rows = src.length;
    const cols = src[0].length;
    for (let i = 0; i < rows; i++)
        for (let j = 0; j < cols; j++)
            dst[r0 + i][c0 + j] = src[i][j];
}

/**
 * Create an nRows×nCols zero matrix.
 *
 * @param {number} nRows
 * @param {number} nCols
 * @returns {number[][]}
 */
function matZeroNM(nRows, nCols) {
    const Z = new Array(nRows);
    for (let i = 0; i < nRows; i++) Z[i] = new Array(nCols).fill(0);
    return Z;
}

// ---------------------------------------------------------------------------
// Preintegration API
// ---------------------------------------------------------------------------

/**
 * Create a fresh preintegration accumulator.
 *
 * The returned object is a plain data bag that `preintegrateStep` mutates
 * in place.
 *
 * @returns {object} Accumulator with the fields described above.
 */
export function createPreintegrator() {
    return {
        /** @type {number[][]} cumulative rotation delta (3×3) */
        dR: mat3Identity(),
        /** @type {number[]} cumulative velocity delta [3] */
        dv: [0, 0, 0],
        /** @type {number[]} cumulative position delta [3] */
        dp: [0, 0, 0],
        /** @type {number[][]} 9×9 covariance [δθ(3), δv(3), δp(3)] */
        cov: matIdentity(9),
        /** @type {number} total accumulated time */
        dtSum: 0,
        /** @type {number[][]|null} Jacobian of ΔR w.r.t. gyro bias (3×3) */
        dRdBg: null,
        /** @type {number[][]|null} Jacobian of Δv w.r.t. gyro bias (3×3) */
        dVdBg: null,
        /** @type {number[][]|null} Jacobian of Δv w.r.t. accelerometer bias (3×3) */
        dVdBa: null,
        /** @type {number[][]|null} Jacobian of Δp w.r.t. gyro bias (3×3) */
        dPdBg: null,
        /** @type {number[][]|null} Jacobian of Δp w.r.t. accelerometer bias (3×3) */
        dPdBa: null,

        // Internal state
        _first: true,
        _prevOmega: [0, 0, 0],
        _prevAccel: [0, 0, 0],
    };
}

/**
 * Accumulate one IMU sample using mid-point (Forster) integration.
 *
 * On the very first call the sample is stored and no delta is
 * accumulated — the first *actual* integration step uses samples 0
 * and 1.
 *
 * @param {object}   preint     Preintegration accumulator from `createPreintegrator()`
 * @param {number[]} omega      Current gyroscope reading [ωx, ωy, ωz] body FRD (rad/s)
 * @param {number[]} accel      Current raw accelerometer reading [ax, ay, az] body FRD (m/s², includes gravity, reads +g on +Z at rest)
 * @param {number}   dt         Time step from previous to current sample (s)
 * @param {number[]} [prevOmega] Previous gyro reading (optional; function tracks internally)
 * @param {number[]} [prevAccel] Previous raw accelerometer reading (optional)
 * @param {number[]} [b_g]      Gyroscope bias [3] rad/s (default zeros)
 * @param {number[]} [b_a]      Accelerometer bias [3] m/s² (default zeros)
 */
export function preintegrateStep(preint, omega, accel, dt, prevOmega, prevAccel, b_g, b_a) {
    const bg = b_g || [0, 0, 0];
    const ba = b_a || [0, 0, 0];

    if (preint._first) {
        preint._prevOmega = [omega[0], omega[1], omega[2]];
        preint._prevAccel = [accel[0], accel[1], accel[2]];
        preint._first = false;
        return;
    }

    const pOmega = preint._prevOmega;
    const pAccel = preint._prevAccel;

    // ---- mid-point de-biased IMU ------------------------------------------
    const omMid = [
        0.5 * (omega[0] + pOmega[0]) - bg[0],
        0.5 * (omega[1] + pOmega[1]) - bg[1],
        0.5 * (omega[2] + pOmega[2]) - bg[2],
    ];
    // Specific force = negative of de-biased sensor reading (MEMS convention)
    const axRaw = 0.5 * (accel[0] + pAccel[0]) - ba[0];
    const ayRaw = 0.5 * (accel[1] + pAccel[1]) - ba[1];
    const azRaw = 0.5 * (accel[2] + pAccel[2]) - ba[2];
    const acMid = [-axRaw, -ayRaw, -azRaw];

    // ---- SO(3) increment --------------------------------------------------
    const phi = [omMid[0] * dt, omMid[1] * dt, omMid[2] * dt];
    const theta = Math.sqrt(phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2]);
    const dR_i = theta < 1e-10 ? mat3Identity() : expRodrigues(phi);

    // ---- save pre-update values for delta & covariance propagation --------
    const dR_prev = mat3Copy(preint.dR);
    const dv_prev = [preint.dv[0], preint.dv[1], preint.dv[2]];

    // ---- accumulated delta updates ----------------------------------------
    // ΔR ← ΔR · ΔR_i
    preint.dR = mat3Mul(dR_prev, dR_i);

    const aWorld = mat3VecMul(dR_prev, acMid);

    // Δv ← Δv + ΔR_prev · a_mid · dt
    preint.dv[0] += aWorld[0] * dt;
    preint.dv[1] += aWorld[1] * dt;
    preint.dv[2] += aWorld[2] * dt;

    // Δp ← Δp + Δv_prev · dt + ½ · ΔR_prev · a_mid · dt²
    const halfDt2 = 0.5 * dt * dt;
    preint.dp[0] += dv_prev[0] * dt + aWorld[0] * halfDt2;
    preint.dp[1] += dv_prev[1] * dt + aWorld[1] * halfDt2;
    preint.dp[2] += dv_prev[2] * dt + aWorld[2] * halfDt2;

    // ---- covariance propagation (9×9) -------------------------------------
    const dR_iT = mat3Transpose(dR_i);

    // ---- A matrix (9×9) --------------------------------------------------
    // A = I9
    // A[0:3,0:3] = ΔR_i^T
    // A[3:6,0:3] = −ΔR_prev · skew(a_mid) · dt
    // A[6:9,0:3] = −½ · ΔR_prev · skew(a_mid) · dt²
    // A[6:9,3:6] = I3 · dt
    const A = matIdentity(9);
    matSetBlock(A, 0, 0, dR_iT);

    const skAc = skew(acMid);
    const Rsk = mat3Mul(dR_prev, skAc);

    matSetBlock(A, 3, 0, mat3Scale(Rsk, -dt));
    matSetBlock(A, 6, 0, mat3Scale(Rsk, -halfDt2));

    const i3dt = [[dt, 0, 0], [0, dt, 0], [0, 0, dt]];
    matSetBlock(A, 6, 3, i3dt);

    // ---- B matrix (9×6) --------------------------------------------------
    // B[0:3,3:6] = Jr(φ) · dt
    // B[3:6,0:3] = ΔR_prev · dt
    // B[6:9,0:3] = ½ · ΔR_prev · dt²
    const B = matZeroNM(9, 6);

    const Jr = jrSimplified(phi);
    matSetBlock(B, 0, 3, mat3Scale(Jr, dt));
    matSetBlock(B, 3, 0, mat3Scale(dR_prev, dt));
    matSetBlock(B, 6, 0, mat3Scale(dR_prev, halfDt2));

    // ---- noise covariance Q6 = diag(σ_a² I3, σ_g² I3) --------------------
    const qDiag = [
        SIGMA_ACC * SIGMA_ACC,
        SIGMA_ACC * SIGMA_ACC,
        SIGMA_ACC * SIGMA_ACC,
        SIGMA_GYRO * SIGMA_GYRO,
        SIGMA_GYRO * SIGMA_GYRO,
        SIGMA_GYRO * SIGMA_GYRO,
    ];

    // cov ← A · cov · A^T + B · Q · B^T
    preint.cov = matAdd(
        matMulABAT(A, preint.cov),
        matMulDiagBT(B, qDiag),
    );

    preint.dtSum += dt;

    // ---- store for next call ----------------------------------------------
    preint._prevOmega = [omega[0], omega[1], omega[2]];
    preint._prevAccel = [accel[0], accel[1], accel[2]];
}

/**
 * Apply the preintegrated delta to predict state at the next keyframe.
 *
 *   R_j   = R_i · ΔR
 *   v_j   = R_i · Δv  +  v_i  +  g_world · Δt_sum
 *   p_j   = R_i · Δp  +  p_i  +  v_i · Δt_sum  +  ½ · g_world · Δt_sum²
 *
 * @param {object}     preint  Preintegration accumulator (after all steps)
 * @param {number[]}   p_i     Position of keyframe i in world NED [n, e, d] (m)
 * @param {number[]}   v_i     Velocity of keyframe i in world NED [vn, ve, vd] (m/s)
 * @param {number[][]} R_i     Rotation matrix of keyframe i (body→world, 3×3)
 * @returns {{p_j: number[], v_j: number[], q_j: number[]}} Predicted state at keyframe j
 */
export function predictState(preint, p_i, v_i, R_i) {
    const dtSum = preint.dtSum;

    // R_j = R_i · ΔR
    const R_j = mat3Mul(R_i, preint.dR);

    const dVw = mat3VecMul(R_i, preint.dv);
    const dPw = mat3VecMul(R_i, preint.dp);

    const halfDt2 = 0.5 * dtSum * dtSum;

    const v_j = [
        dVw[0] + v_i[0] + G_WORLD[0] * dtSum,
        dVw[1] + v_i[1] + G_WORLD[1] * dtSum,
        dVw[2] + v_i[2] + G_WORLD[2] * dtSum,
    ];

    const p_j = [
        dPw[0] + p_i[0] + v_i[0] * dtSum + G_WORLD[0] * halfDt2,
        dPw[1] + p_i[1] + v_i[1] * dtSum + G_WORLD[1] * halfDt2,
        dPw[2] + p_i[2] + v_i[2] * dtSum + G_WORLD[2] * halfDt2,
    ];

    const q_j = rotToQuat(R_j);

    return { p_j, v_j, q_j };
}
