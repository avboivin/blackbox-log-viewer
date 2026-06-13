/**
 * Error-State Kalman Filter with configurable state dimension.
 *
 * Base 9-state: [δp(3), δv(3), δθ(3)]
 * Extended adds magnetic field states m_earth(3) world + m_body(3) body (total 15).
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

function matMul(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n).fill(0);
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            if (aik === 0) continue;
            for (let j = 0; j < n; j++) C[i][j] += aik * B[k][j];
        }
    }
    return C;
}

function matAdd(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n);
        for (let j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
    }
    return C;
}

function matTranspose(A) {
    const n = A.length;
    const T = new Array(n);
    for (let i = 0; i < n; i++) {
        T[i] = new Array(n);
        for (let j = 0; j < n; j++) T[i][j] = A[j][i];
    }
    return T;
}

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
            if (Math.abs(aug[r][i]) > maxVal) { maxVal = Math.abs(aug[r][i]); maxRow = r; }
        }
        if (maxRow !== i) { const tmp = aug[i]; aug[i] = aug[maxRow]; aug[maxRow] = tmp; }
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
// F and Q builders
// ---------------------------------------------------------------------------

const IDX_ME = 9, IDX_MB = 12;

// Global (world-frame) error-state transition for the [δp, δv, δθ] block.
// Attitude error is defined GLOBALLY:  q_true = δq(δθ_world) ⊗ q̂  (see 06 §1).
// In this convention the deterministic attitude-error propagation is the
// identity (F_θθ = I — no skew(ω) term; gyro error enters only via Q), and the
// position/velocity couple to attitude through the WORLD specific force R·f:
//   F_pθ = −skew(R·f)·½dt²,  F_vθ = −skew(R·f)·dt.
// (The local/body form −R·skew(f) and the I−skew(ω)dt attitude block are WRONG
//  here — they silently corrupt any non-level, rotating segment. FD-verified.)
function buildTransition(dim, q, sfAccel, dt) {
    const F = matIdentity(dim);
    const R = quatToRot(q);
    const [sx, sy, sz] = sfAccel;
    const dt2h = 0.5 * dt * dt;

    // World-frame specific force  fw = R · f_body
    const fwx = R[0][0]*sx + R[0][1]*sy + R[0][2]*sz;
    const fwy = R[1][0]*sx + R[1][1]*sy + R[1][2]*sz;
    const fwz = R[2][0]*sx + R[2][1]*sy + R[2][2]*sz;

    // skew(fw)
    const sr00 = 0,    sr01 = -fwz, sr02 = fwy;
    const sr10 = fwz,  sr11 = 0,    sr12 = -fwx;
    const sr20 = -fwy, sr21 = fwx,  sr22 = 0;

    // δp ← δv
    F[0][3]=dt; F[1][4]=dt; F[2][5]=dt;
    // δp ← δθ  = −skew(fw)·½dt²
    F[0][6]=-sr00*dt2h; F[0][7]=-sr01*dt2h; F[0][8]=-sr02*dt2h;
    F[1][6]=-sr10*dt2h; F[1][7]=-sr11*dt2h; F[1][8]=-sr12*dt2h;
    F[2][6]=-sr20*dt2h; F[2][7]=-sr21*dt2h; F[2][8]=-sr22*dt2h;
    // δv ← δθ  = −skew(fw)·dt
    F[3][6]=-sr00*dt; F[3][7]=-sr01*dt; F[3][8]=-sr02*dt;
    F[4][6]=-sr10*dt; F[4][7]=-sr11*dt; F[4][8]=-sr12*dt;
    F[5][6]=-sr20*dt; F[5][7]=-sr21*dt; F[5][8]=-sr22*dt;
    // δθ ← δθ  = I  (already set by matIdentity; gyro error enters via Q)

    return F;
}

function buildProcessNoise(dim, sigmaAcc, sigmaGyro, dt) {
    const Q = new Array(dim);
    for (let i = 0; i < dim; i++) Q[i] = new Array(dim).fill(0);
    const sa2 = sigmaAcc * sigmaAcc;
    const sg2 = sigmaGyro * sigmaGyro;
    const dt2 = dt * dt;
    const pNoise = sa2 * dt2 * dt2 / 4;
    const vNoise = sa2 * dt2;
    const pvNoise = sa2 * dt2 * dt / 2;
    const gNoise = sg2 * dt2;
    Q[0][0]=Q[1][1]=Q[2][2]=pNoise;
    Q[3][3]=Q[4][4]=Q[5][5]=vNoise;
    Q[0][3]=Q[3][0]=Q[1][4]=Q[4][1]=Q[2][5]=Q[5][2]=pvNoise;
    Q[6][6]=Q[7][7]=Q[8][8]=gNoise;
    return Q;
}

function symmetryForce(P) {
    const n = P.length;
    for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++) {
            const avg = (P[i][j] + P[j][i]) * 0.5;
            P[i][j] = avg; P[j][i] = avg;
        }
}

function varianceFloor(P, minVal = 1e-6) {
    for (let i = 0; i < P.length; i++)
        if (P[i][i] < minVal) P[i][i] = minVal;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * @param {object} opts
 * @param {number[]} opts.p0
 * @param {number[]} opts.v0
 * @param {number[]} opts.q0
 * @param {number} [opts.sigmaPos=5]
 * @param {number} [opts.sigmaVel=2]
 * @param {number} [opts.sigmaAtt=0.2]
 * @param {number[]} [opts.mEarth0] - initial earth field NED [Gauss], enables 15-state
 * @param {number[]} [opts.mBody0] - initial body hard-iron [Gauss]
 * @param {number} [opts.sigmaMagEarth=0.05]
 * @param {number} [opts.sigmaMagBody=0.02]
 */
export function createEskf({ p0, v0, q0, sigmaPos = 5, sigmaVel = 2, sigmaAtt = 0.2, mEarth0, mBody0, sigmaMagEarth = 0.05, sigmaMagBody = 0.02 }) {
    const hasMag = mEarth0 != null;
    const dim = hasMag ? 15 : 9;

    const P = matIdentity(dim);
    P[0][0] = P[1][1] = P[2][2] = sigmaPos * sigmaPos;
    P[3][3] = P[4][4] = P[5][5] = sigmaVel * sigmaVel;
    P[6][6] = P[7][7] = P[8][8] = sigmaAtt * sigmaAtt;

    const me = hasMag ? mEarth0.slice() : null;
    const mb = hasMag ? (mBody0 ? mBody0.slice() : [0, 0, 0]) : null;

    if (hasMag) {
        P[9][9] = P[10][10] = P[11][11] = sigmaMagEarth * sigmaMagEarth;
        P[12][12] = P[13][13] = P[14][14] = sigmaMagBody * sigmaMagBody;
    }

    return {
        dim,
        p: p0.slice(),
        v: v0.slice(),
        q: q0.slice(),
        mEarth: me,
        mBody: mb,
        P,
        sigmaAcc: 0.35,
        sigmaGyro: 0.015,
        sigmaMagEarthRW: 1e-3,
        sigmaMagBodyRW: 1e-4,
    };
}

/**
 * Predict step. Returns the transition matrix F used (for RTS smoother).
 *
 * @returns {{ F: number[][] }} the error-state transition matrix (dim×dim)
 */
export function eskfPredict(eskf, omega, accel, dt) {
    const { dim } = eskf;

    const sfX = -accel[0], sfY = -accel[1], sfZ = -accel[2];
    const F = buildTransition(dim, eskf.q, [sfX, sfY, sfZ], dt);
    const Q = buildProcessNoise(dim, eskf.sigmaAcc, eskf.sigmaGyro, dt);

    const next = strapdownPropagate(omega, accel, eskf.q, eskf.v, eskf.p, dt);
    eskf.p = next.p;
    eskf.v = next.v;
    eskf.q = next.q;

    // Add random-walk noise for magnetic field states
    if (dim >= 15) {
        const meRw = eskf.sigmaMagEarthRW * eskf.sigmaMagEarthRW * dt;
        const mbRw = eskf.sigmaMagBodyRW * eskf.sigmaMagBodyRW * dt;
        Q[9][9] += meRw; Q[10][10] += meRw; Q[11][11] += meRw;
        Q[12][12] += mbRw; Q[13][13] += mbRw; Q[14][14] += mbRw;
    }

    const FP = matMul(F, eskf.P);
    const FPFt = matMul(FP, matTranspose(F));
    eskf.P = matAdd(FPFt, Q);
    symmetryForce(eskf.P);

    return { F };
}

/**
 * Update step with a measurement factor.
 * The factor's H rows must match eskf.dim in length.
 */
export function eskfUpdate(eskf, factor, z, gate = 3.0) {
    const { dim } = eskf;
    const x = { p: eskf.p, v: eskf.v, q: eskf.q, mEarth: eskf.mEarth, mBody: eskf.mBody };

    const r = factor.residual(z, x);
    let H = factor.H;
    const R = factor.R;
    const m = r.length;

    // Pad H rows to dim if shorter (for 9→15 state extensions)
    if (H[0].length < dim) {
        H = H.map((row) => {
            const padded = new Array(dim).fill(0);
            for (let i = 0; i < row.length; i++) padded[i] = row[i];
            return padded;
        });
    }

    // S = H·P·Hᵀ + R
    const PHt = new Array(dim);
    for (let i = 0; i < dim; i++) {
        PHt[i] = new Array(m).fill(0);
        for (let k = 0; k < dim; k++) {
            const pik = eskf.P[i][k];
            if (pik === 0) continue;
            for (let j = 0; j < m; j++) PHt[i][j] += pik * H[j][k];
        }
    }

    const S = new Array(m);
    for (let i = 0; i < m; i++) {
        S[i] = new Array(m).fill(0);
        for (let j = 0; j < m; j++) {
            for (let k = 0; k < dim; k++) S[i][j] += H[i][k] * PHt[k][j];
            S[i][j] += R[i][j];
        }
    }

    const S_inv = matInvertSym(S);
    let mahal = 0;
    for (let i = 0; i < m; i++)
        for (let j = 0; j < m; j++) mahal += r[i] * S_inv[i][j] * r[j];
    if (mahal > gate * gate * m) return false;

    // Kalman gain
    const K = new Array(dim);
    for (let i = 0; i < dim; i++) {
        K[i] = new Array(m);
        for (let j = 0; j < m; j++) {
            let s = 0;
            for (let k = 0; k < m; k++) s += PHt[i][k] * S_inv[k][j];
            K[i][j] = s;
        }
    }

    const dx = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++)
        for (let j = 0; j < m; j++) dx[i] += K[i][j] * r[j];

    // Inject into nominal state
    eskf.p[0] += dx[0]; eskf.p[1] += dx[1]; eskf.p[2] += dx[2];
    eskf.v[0] += dx[3]; eskf.v[1] += dx[4]; eskf.v[2] += dx[5];

    const dtheta = [dx[6], dx[7], dx[8]];
    const dthetaNorm = Math.sqrt(dtheta[0]**2 + dtheta[1]**2 + dtheta[2]**2);
    if (dthetaNorm > 1e-12) {
        const axis = [dtheta[0]/dthetaNorm, dtheta[1]/dthetaNorm, dtheta[2]/dthetaNorm];
        const dq = quatFromAxisAngle(axis, dthetaNorm);
        const newQ = quatMultiply(dq, eskf.q);
        const nq = Math.sqrt(newQ[0]**2+newQ[1]**2+newQ[2]**2+newQ[3]**2);
        eskf.q = [newQ[0]/nq, newQ[1]/nq, newQ[2]/nq, newQ[3]/nq];
    }

    if (eskf.mEarth) {
        eskf.mEarth[0] += dx[9]; eskf.mEarth[1] += dx[10]; eskf.mEarth[2] += dx[11];
    }
    if (eskf.mBody) {
        eskf.mBody[0] += dx[12]; eskf.mBody[1] += dx[13]; eskf.mBody[2] += dx[14];
    }

    // Joseph form
    const I_KH = matIdentity(dim);
    for (let i = 0; i < dim; i++)
        for (let j = 0; j < dim; j++)
            for (let k = 0; k < m; k++) I_KH[i][j] -= K[i][k] * H[k][j];

    const IKH_P = matMul(I_KH, eskf.P);
    const IKH_P_IKHt = matMul(IKH_P, matTranspose(I_KH));

    const KRKt = new Array(dim);
    for (let i = 0; i < dim; i++) {
        KRKt[i] = new Array(dim).fill(0);
        for (let j = 0; j < dim; j++)
            for (let ki = 0; ki < m; ki++)
                for (let kj = 0; kj < m; kj++)
                    KRKt[i][j] += K[i][ki] * R[ki][kj] * K[j][kj];
    }

    eskf.P = matAdd(IKH_P_IKHt, KRKt);
    symmetryForce(eskf.P);
    varianceFloor(eskf.P);
    return true;
}

export function eskfGetState(eskf) {
    return {
        p: eskf.p.slice(),
        v: eskf.v.slice(),
        q: eskf.q.slice(),
        mEarth: eskf.mEarth ? eskf.mEarth.slice() : null,
        mBody: eskf.mBody ? eskf.mBody.slice() : null,
        sigmaPos: Math.sqrt(Math.max(0, (eskf.P[0][0]+eskf.P[1][1]+eskf.P[2][2])/3)),
        sigmaAtt: Math.sqrt(Math.max(0, (eskf.P[6][6]+eskf.P[7][7]+eskf.P[8][8])/3))*(180/Math.PI),
    };
}

export { IDX_ME, IDX_MB };
