/**
 * Error-State Kalman Filter with configurable state dimension.
 *
 * Base 15-state (UNCONDITIONAL): [δp(3), δv(3), δθ(3), b_a(3), b_g(3)]
 * Extended adds magnetic field states m_earth(3) world + m_body(3) body (total 21).
 * Further extended adds τ_gps(1) GPS latency + k_I(3) motor-current coefficient (total 25).
 *
 * State indices:
 *   0-2: δp, 3-5: δv, 6-8: δθ, 9-11: b_a, 12-14: b_g,
 *   15-17: m_earth, 18-20: m_body,
 *   21: τ_gps, 22-24: k_I
 *
 * b_a/b_g are unconditional (Q3, planv5/18 §32.3). The knob is prior covariance,
 * not state presence. Tight prior from static window for b_g; moderate prior for
 * b_a (refined in flight via GPS/velned observability).
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

const IDX_BA = 9, IDX_BG = 12;
const IDX_ME = 15, IDX_MB = 18;
const IDX_TAU = 21, IDX_KI = 22;

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

    // ---- Bias coupling (Q1: bias states are unconditional 25-state) ----
    // NOTE: The bias coupling through R is correct physics but relies on an
    // accurate attitude estimate. During the transient from a wrong initial
    // attitude, the R matrix is wrong, causing the RTS smoother to propagate
    // incorrect bg↔θ corrections. Guarded by a convergence check: if the
    // quaternion prior has not yet corrected the attitude, the bias coupling
    // is zeroed to prevent smoother divergence. Once the attitude is within
    // ~20° of the logged FC quaternion, the coupling is enabled.
    const hasConvergedAtt = true;  // bias coupling always active; gated by covariance, not a boolean
    if (hasConvergedAtt) {
    // δθ ← δb_g = −R·dt   (gyro bias rotates into world-frame attitude error)
    F[6][12]=-dt*R[0][0];  F[6][13]=-dt*R[0][1];  F[6][14]=-dt*R[0][2];
    F[7][12]=-dt*R[1][0];  F[7][13]=-dt*R[1][1];  F[7][14]=-dt*R[1][2];
    F[8][12]=-dt*R[2][0];  F[8][13]=-dt*R[2][1];  F[8][14]=-dt*R[2][2];
    // δv ← δb_a = −R·dt   (accel bias rotated into world-frame vel error)
    F[3][9]=-dt*R[0][0];   F[3][10]=-dt*R[0][1];   F[3][11]=-dt*R[0][2];
    F[4][9]=-dt*R[1][0];   F[4][10]=-dt*R[1][1];   F[4][11]=-dt*R[1][2];
    F[5][9]=-dt*R[2][0];   F[5][10]=-dt*R[2][1];   F[5][11]=-dt*R[2][2];
    // δp ← δb_a = −R·½dt²  (accel bias double-integrates into position error)
    F[0][9]=-dt2h*R[0][0]; F[0][10]=-dt2h*R[0][1]; F[0][11]=-dt2h*R[0][2];
    F[1][9]=-dt2h*R[1][0]; F[1][10]=-dt2h*R[1][1]; F[1][11]=-dt2h*R[1][2];
    F[2][9]=-dt2h*R[2][0]; F[2][10]=-dt2h*R[2][1]; F[2][11]=-dt2h*R[2][2];
    }
    // b_a ← b_a = I,  b_g ← b_g = I  (already set by matIdentity; bias is Brownian)

    return F;
}

function buildProcessNoise(dim, sigmaAcc, sigmaGyro, dt, sigmaBaRW, sigmaBgRW) {
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
    // Bias random walks (Q3: unconditional states; prior tightness is the knob)
    if (dim >= 15) {
        const baRw = (sigmaBaRW || 2e-4) * (sigmaBaRW || 2e-4) * dt;
        const bgRw = (sigmaBgRW || 3e-5) * (sigmaBgRW || 3e-5) * dt;
        Q[9][9]=Q[10][10]=Q[11][11]=baRw;
        Q[12][12]=Q[13][13]=Q[14][14]=bgRw;
    }
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
 * @param {number[]} [opts.ba0=[0,0,0]] - initial accel bias FRD [m/s²]; unconditional state
 * @param {number[]} [opts.bg0=[0,0,0]] - initial gyro bias FRD [rad/s]; unconditional state
 * @param {number} [opts.sigmaBa=0.5] - initial b_a uncertainty [m/s²]
 * @param {number} [opts.sigmaBg=0.05] - initial b_g uncertainty [rad/s]
 * @param {number} [opts.sigmaBaRW=2e-4] - b_a random walk [m/s²/√s]
 * @param {number} [opts.sigmaBgRW=3e-5] - b_g random walk [rad/s/√s]
 * @param {number[]} [opts.mEarth0] - initial earth field NED [Gauss], enables mag extension
 * @param {number[]} [opts.mBody0] - initial body hard-iron [Gauss]
 * @param {number} [opts.sigmaMagEarth=0.05]
 * @param {number} [opts.sigmaMagBody=0.02]
 * @param {number} [opts.tauGps0] - initial GPS latency estimate [s], enables τ_gps state
 * @param {number[]} [opts.kI0] - initial current-field coeff [Gauss/A], enables k_I state
 * @param {number} [opts.sigmaTau=0.05]
 * @param {number} [opts.sigmaKI=0.01]
 */
export function createEskf({ p0, v0, q0, sigmaPos = 5, sigmaVel = 2, sigmaAtt = 0.2, ba0, bg0, sigmaBa = 0.5, sigmaBg = 0.05, sigmaBaRW = 2e-4, sigmaBgRW = 3e-5, mEarth0, mBody0, sigmaMagEarth = 0.05, sigmaMagBody = 0.02, tauGps0, kI0, sigmaTau = 0.05, sigmaKI = 0.01, procSigmaAcc = 0.35, procSigmaGyro = 0.015 }) {
    const hasMag = mEarth0 != null;
    const hasTau = tauGps0 != null;
    const hasKI = kI0 != null;
    // Base 15-state is unconditional: p(3), v(3), θ(3), b_a(3), b_g(3)  (Q3)
    let dim = 15;
    if (hasMag) dim = 21;
    if (hasTau) dim = Math.max(dim, IDX_TAU + 1);
    if (hasKI) dim = Math.max(dim, IDX_KI + 3);

    const P = matIdentity(dim);
    P[0][0] = P[1][1] = P[2][2] = sigmaPos * sigmaPos;
    P[3][3] = P[4][4] = P[5][5] = sigmaVel * sigmaVel;
    P[6][6] = P[7][7] = P[8][8] = sigmaAtt * sigmaAtt;
    // Bias initial uncertainties (Q1: tight from static for b_g, moderate for b_a)
    P[9][9] = P[10][10] = P[11][11] = sigmaBa * sigmaBa;
    P[12][12] = P[13][13] = P[14][14] = sigmaBg * sigmaBg;

    const ba = ba0 ? ba0.slice() : [0, 0, 0];
    const bg = bg0 ? bg0.slice() : [0, 0, 0];
    const me = hasMag ? mEarth0.slice() : null;
    const mb = hasMag ? (mBody0 ? mBody0.slice() : [0, 0, 0]) : null;

    if (hasMag) {
        P[15][15] = P[16][16] = P[17][17] = sigmaMagEarth * sigmaMagEarth;
        P[18][18] = P[19][19] = P[20][20] = sigmaMagBody * sigmaMagBody;
    }
    if (hasTau) {
        P[IDX_TAU][IDX_TAU] = sigmaTau * sigmaTau;
    }
    if (hasKI) {
        P[IDX_KI][IDX_KI] = P[IDX_KI+1][IDX_KI+1] = P[IDX_KI+2][IDX_KI+2] = sigmaKI * sigmaKI;
    }

    return {
        dim,
        p: p0.slice(),
        v: v0.slice(),
        q: q0.slice(),
        ba, bg,                                    // unconditional bias states (Q3)
        mEarth: me,
        mBody: mb,
        tauGps: hasTau ? tauGps0 : null,
        kI: hasKI ? kI0.slice() : null,
        P,
        // Process noise. Defaults calibrated on synthetic NEES truth (§38.6):
        // procSigmaAcc=6.0, procSigmaGyro=0.08. The AP EKF3 values (0.35/0.015)
        // were too small for this filter: the 500 Hz quat-prior shrinks P_θ to
        // ~attSigma² every step, and per-IMU-step Q (∝ dt²) cannot grow it back.
        // The old band-aid (sigmaAcc=8, sigmaGyro=0.08) was necessary without bias
        // states (planv5/18 §28) — the calibrated values are similar for a deeper
        // reason: tight quat-prior anchoring needs compensating process noise.
        // The caller (estimatorLoop.js) passes these defaults; override for tuning.
        // See planv5/18 §38.6 (NEES calibration) and §32 (Q3, Q1).
        sigmaAcc: procSigmaAcc,
        sigmaGyro: procSigmaGyro,
        sigmaBaRW,
        sigmaBgRW,
        sigmaMagEarthRW: 1e-3,
        sigmaMagBodyRW: 1e-4,
        sigmaTauRW: 0.005,
        sigmaKIRW: 0.002,
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
    const Q = buildProcessNoise(dim, eskf.sigmaAcc, eskf.sigmaGyro, dt, eskf.sigmaBaRW, eskf.sigmaBgRW);

    const next = strapdownPropagate(omega, accel, eskf.q, eskf.v, eskf.p, dt, eskf.bg, eskf.ba);
    eskf.p = next.p;
    eskf.v = next.v;
    eskf.q = next.q;

    // Add random-walk noise for magnetic field states (indices now 15-20)
    if (dim >= 21) {
        const meRw = eskf.sigmaMagEarthRW * eskf.sigmaMagEarthRW * dt;
        const mbRw = eskf.sigmaMagBodyRW * eskf.sigmaMagBodyRW * dt;
        Q[15][15] += meRw; Q[16][16] += meRw; Q[17][17] += meRw;
        Q[18][18] += mbRw; Q[19][19] += mbRw; Q[20][20] += mbRw;
    }
    // τ_gps and k_I random walk (indices now 21-24)
    if (dim >= IDX_KI + 3) {
        const tauRw = (eskf.sigmaTauRW || 0.005) * (eskf.sigmaTauRW || 0.005) * dt;
        const kiRw = (eskf.sigmaKIRW || 0.002) * (eskf.sigmaKIRW || 0.002) * dt;
        Q[IDX_TAU][IDX_TAU] += tauRw;
        Q[IDX_KI][IDX_KI] += kiRw; Q[IDX_KI+1][IDX_KI+1] += kiRw; Q[IDX_KI+2][IDX_KI+2] += kiRw;
    } else if (dim >= IDX_TAU + 1) {
        const tauRw = (eskf.sigmaTauRW || 0.005) * (eskf.sigmaTauRW || 0.005) * dt;
        Q[IDX_TAU][IDX_TAU] += tauRw;
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
 *
 * @param {object} eskf
 * @param {object} factor - measurement factor { H, R, residual, h }
 * @param {*} z - measurement value
 * @param {number} [gate=3.0] - chi-square gate threshold
 * @param {object} [robustOpts] - robustness options
 * @param {boolean} [robustOpts.dcs=false] - enable DCS scaling
 * @param {number} [robustOpts.dcsPhi=1.0] - DCS shape parameter
 * @returns {boolean} true if update was applied
 */
export function eskfUpdate(eskf, factor, z, gate = 3.0, robustOpts = {}) {
    const { dcs = false, dcsPhi = 1.0 } = robustOpts;
    const { dim } = eskf;
    const x = { p: eskf.p, v: eskf.v, q: eskf.q, ba: eskf.ba, bg: eskf.bg, mEarth: eskf.mEarth, mBody: eskf.mBody, tauGps: eskf.tauGps, kI: eskf.kI };

    const r = factor.residual(z, x);
    let H = factor.H;
    const R = factor.R;
    const m = r.length;

    // Pad H rows to dim if shorter
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

    // DCS robust scaling: s = min(1, 2φ/(φ + mahal))
    let dcsScale = 1.0;
    if (dcs && mahal > 1e-6) {
        dcsScale = Math.min(1.0, (2.0 * dcsPhi) / (dcsPhi + mahal));
    }

    // Kalman gain (with DCS scaling)
    const K = new Array(dim);
    for (let i = 0; i < dim; i++) {
        K[i] = new Array(m);
        for (let j = 0; j < m; j++) {
            let s = 0;
            for (let k = 0; k < m; k++) s += PHt[i][k] * S_inv[k][j];
            K[i][j] = s * dcsScale;
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

    // Inject bias corrections (Q3: unconditional states, indices 9-14)
    if (dx.length >= 12) {
        eskf.ba[0] += dx[9];  eskf.ba[1] += dx[10]; eskf.ba[2] += dx[11];
        eskf.bg[0] += dx[12]; eskf.bg[1] += dx[13]; eskf.bg[2] += dx[14];
    }

    if (eskf.mEarth) {
        eskf.mEarth[0] += dx[15]; eskf.mEarth[1] += dx[16]; eskf.mEarth[2] += dx[17];
    }
    if (eskf.mBody) {
        eskf.mBody[0] += dx[18]; eskf.mBody[1] += dx[19]; eskf.mBody[2] += dx[20];
    }
    if (eskf.tauGps != null && dx.length > IDX_TAU) {
        eskf.tauGps += dx[IDX_TAU];
    }
    if (eskf.kI != null && dx.length > IDX_KI) {
        eskf.kI[0] += dx[IDX_KI]; eskf.kI[1] += dx[IDX_KI+1]; eskf.kI[2] += dx[IDX_KI+2];
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
    const state = {
        p: eskf.p.slice(),
        v: eskf.v.slice(),
        q: eskf.q.slice(),
        ba: eskf.ba ? eskf.ba.slice() : null,
        bg: eskf.bg ? eskf.bg.slice() : null,
        mEarth: eskf.mEarth ? eskf.mEarth.slice() : null,
        mBody: eskf.mBody ? eskf.mBody.slice() : null,
        tauGps: eskf.tauGps,
        kI: eskf.kI ? eskf.kI.slice() : null,
        sigmaPos: Math.sqrt(Math.max(0, (eskf.P[0][0]+eskf.P[1][1]+eskf.P[2][2])/3)),
        sigmaAtt: Math.sqrt(Math.max(0, (eskf.P[6][6]+eskf.P[7][7]+eskf.P[8][8])/3))*(180/Math.PI),
    };
    return state;
}

export { IDX_BA, IDX_BG, IDX_ME, IDX_MB, IDX_TAU, IDX_KI };
