/**
 * Measurement models (factors) for a drone body-pose ESKF estimator.
 *
 * Error state δx = [δp(3), δv(3), δθ(3), b_a(3), b_g(3), m_earth(3), m_body(3), τ(1), k_I(3)]
 *   δp  = position error (NED, m)
 *   δv  = velocity error (NED, m/s)
 *   δθ  = attitude error (rotation vector in world frame, rad)
 *   b_a = accelerometer bias (FRD, m/s²) — unconditional (Q3)
 *   b_g = gyroscope bias (FRD, rad/s) — unconditional (Q3)
 *
 * Nominal state x = { p, v, q, ba, bg, mEarth?, mBody?, tauGps?, kI? }
 *   p = position in NED (m)
 *   v = velocity in NED (m/s)
 *   q = [w,x,y,z] scalar-first, body(FRD) → world(NED)
 *
 * Gravity: [0, 0, +9.80665] in NED
 *
 * State indices (Q3: 25-state):
 *   0-2: δp, 3-5: δv, 6-8: δθ, 9-11: b_a, 12-14: b_g,
 *   15-17: m_earth, 18-20: m_body, 21: τ_gps, 22-24: k_I
 *
 * Jacobian row layout: each row is length matched to the state dimension.
 * Each factory returns an object with:
 *   h(x)         – measurement prediction
 *   H            – 1D N-element array per measurement row
 *   R            – noise covariance matrix
 *   residual(z,x)– computes r = z − h(x)
 */

const GRAVITY_MAG = 9.80665;

// --- Quaternion math helpers --------------------------------------------------

function quatMultiply(a, b) {
  // a, b are [w,x,y,z] (scalar-first)
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

function quatConjugate(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

function quatToRotMat(q) {
  // Canonical body(FRD)→world(NED) rotation R, 3×3 row-major flat.
  // v_world = R · v_body.  Matches imuMechanization.quatToRot exactly.
  // (Previously this returned Rᵀ — the world→body transpose — which silently
  //  conflicted with the strapdown/injection convention. FD/loop-verified.)
  const [w, x, y, z] = q;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;

  const m = new Array(9);
  m[0] = 1 - 2 * (yy + zz);
  m[1] = 2 * (xy - wz);
  m[2] = 2 * (xz + wy);

  m[3] = 2 * (xy + wz);
  m[4] = 1 - 2 * (xx + zz);
  m[5] = 2 * (yz - wx);

  m[6] = 2 * (xz - wy);
  m[7] = 2 * (yz + wx);
  m[8] = 1 - 2 * (xx + yy);

  return m;  // 3×3 row-major, body→world
}

function logMap(R) {
  // Logarithmic map SO(3) → so(3) rotation vector (3-element), in rad
  const trace = R[0] + R[4] + R[8];
  const cosTheta = (trace - 1) / 2;
  const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)));

  if (Math.abs(theta) < 1e-12) {
    // Small angle approximation: ω ≈ (R − Rᵀ)∨ / 2
    return [
      (R[7] - R[5]) / 2,
      (R[2] - R[6]) / 2,
      (R[3] - R[1]) / 2,
    ];
  }

  const factor = theta / (2 * Math.sin(theta));
  return [
    (R[7] - R[5]) * factor,
    (R[2] - R[6]) * factor,
    (R[3] - R[1]) * factor,
  ];
}

// --- Factory functions --------------------------------------------------------

/**
 * GPS position measurement in NED.
 *
 * @param {{n: number, e: number, d: number}} meas  NED position (m)
 * @param {number} [sigma=2.5]  1σ noise in metres
 */
export function createGpsPositionFactor(meas, sigma = 2.5) {
  const varP = sigma * sigma;

  // h(x) = p
  function h(x) {
    return [x.p[0], x.p[1], x.p[2]];
  }

  // H = [I3, 0_3, 0_3] → each row is 9-element
  const H = [
    [1, 0, 0,  0, 0, 0,  0, 0, 0],
    [0, 1, 0,  0, 0, 0,  0, 0, 0],
    [0, 0, 1,  0, 0, 0,  0, 0, 0],
  ];

  // R = sigma² · I3
  const R = [
    [varP,    0,    0],
    [   0, varP,    0],
    [   0,    0, varP],
  ];

  function residual(z, x) {
    const hp = h(x);
    return [z.n - hp[0], z.e - hp[1], z.d - hp[2]];
  }

  return { h, H, R, residual };
}

/**
 * GPS velocity measurement in NED.
 *
 * @param {{n: number, e: number, d: number}} meas  NED velocity (m/s)
 * @param {number} [sigma=0.5]  1σ noise in m/s
 */
export function createGpsVelocityFactor(meas, sigma = 0.5) {
  const varV = sigma * sigma;

  function h(x) {
    return [x.v[0], x.v[1], x.v[2]];
  }

  const H = [
    [0, 0, 0,  1, 0, 0,  0, 0, 0],
    [0, 0, 0,  0, 1, 0,  0, 0, 0],
    [0, 0, 0,  0, 0, 1,  0, 0, 0],
  ];

  const R = [
    [varV,    0,    0],
    [   0, varV,    0],
    [   0,    0, varV],
  ];

  function residual(z, x) {
    const hv = h(x);
    return [z.n - hv[0], z.e - hv[1], z.d - hv[2]];
  }

  return { h, H, R, residual };
}

/**
 * Barometer altitude measurement.
 *
 * The barometer reads altitude relative to the arming point.
 * baroOffset = GPS_alt_at_arm − baroAlt_at_arm converts baro-relative to MSL-absolute.
 * Measurement model: z ≈ −p_D.
 *
 * The baro reading is relative to the arm point; p_D is the NED down coordinate
 * relative to the origin. Since origin and arm are approximately the same physical
 * location, they match without an offset. Adding baroOffset (~GPS MSL altitude,
 * typically >100 m) creates a constant innovation offset that saturates the 3σ gate
 * and silently rejects all baro measurements — the D coordinate then drifts uncorrected.
 * (planv5/18 §35 — baro-offset bug, 2026-06-15)
 *
 * @param {number} baroAlt     Raw barometer altitude (m, relative to arm point)
 * @param {number} baroOffset  (UNUSED — retained for API compat only)
 * @param {number} [sigma=1.0] 1σ noise in metres
 */
export function createBaroFactor(baroAlt, baroOffset, sigma = 1.0) {
  const varZ = sigma * sigma;

  // h(x) = −p_D (baro reads approximately −p_D when origin ≈ arm point)
  function h(x) {
    return -x.p[2];
  }

  // H = [[0, 0, -1,  0, 0, 0,  0, 0, 0]] — only D component of position
  const H = [[0, 0, -1, 0, 0, 0, 0, 0, 0]];

  const R = [[varZ]];

  function residual(z, x) {
    return [z - h(x)];
  }

  return { h, H, R, residual };
}

/**
 * Quaternion attitude prior (soft prior from FC fused attitude).
 *
 * @param {[number,number,number,number]} qMeas  Measured quaternion [w,x,y,z] scalar-first
 * @param {number} [sigma=0.1]  1σ noise in radians
 */
export function createQuaternionPrior(qMeas, sigma = 0.1) {
  const varQ = sigma * sigma;

  // h(x) — not used directly for residual; residual computed via log map
  function h(x) {
    return x.q.slice();  // return [w,x,y,z]
  }

  // H = [0_3, 0_3, I3]
  const H = [
    [0, 0, 0,  0, 0, 0,  1, 0, 0],
    [0, 0, 0,  0, 0, 0,  0, 1, 0],
    [0, 0, 0,  0, 0, 0,  0, 0, 1],
  ];

  const R = [
    [varQ,    0,    0],
    [   0, varQ,    0],
    [   0,    0, varQ],
  ];

  function residual(z, x) {
    // GLOBAL (world-frame) attitude residual:  r = logMap( R_meas · R_state^T ).
    // With q_true = δq(δθ_world) ⊗ q̂, ∂r/∂δθ_world = −I, so H = [0,0,I] is exact.
    // qMeas ⊗ q_state*  gives R_meas · R_state^T.
    // (The previous 2·logMap(R_meas^T·R_state) was the BODY-frame error with a
    //  spurious factor of 2 — inconsistent with the global injection. FD-verified.)
    const qRel = quatMultiply(qMeas, quatConjugate(x.q));
    const Rrel = quatToRotMat(qRel);
    const omega = logMap(Rrel);
    return [omega[0], omega[1], omega[2]];
  }

  return { h, H, R, residual };
}

/**
 * 3-axis magnetometer measurement (body frame, in Gauss).
 *
 * Measurement model: z_mag = R(q)^T · m_earth + m_body + k_I · I(t)
 *
 * H rows are dimensioned for 19-state (9 base + 6 mag + 1 τ + 3 k_I).
 *
 * @param {number[]} meas   - mag reading [bx,by,bz] body FRD (Gauss)
 * @param {number}   sigma  - measurement noise 1σ (Gauss)
 * @param {number}   [currentAmps=0] - battery current in Amps (drives k_I term)
 * @returns {object} factor
 */
export function createMagFactor(meas, sigma = 0.05, currentAmps = 0) {
    const varM = sigma * sigma;
    const Rnoise = [[varM, 0, 0], [0, varM, 0], [0, 0, varM]];

    let cachedH = null;

    function h(x) {
        const m = quatToRotMat(x.q);
        const me = x.mEarth || [0, 0, 0];
        const mb = x.mBody || [0, 0, 0];
        const kI = x.kI || [0, 0, 0];
        return [
            m[0]*me[0] + m[3]*me[1] + m[6]*me[2] + mb[0] + kI[0] * currentAmps,
            m[1]*me[0] + m[4]*me[1] + m[7]*me[2] + mb[1] + kI[1] * currentAmps,
            m[2]*me[0] + m[5]*me[1] + m[8]*me[2] + mb[2] + kI[2] * currentAmps,
        ];
    }

    function residual(z, x) {
        const m = quatToRotMat(x.q);
        const me = x.mEarth || [0, 0, 0];
        const me0 = me[0], me1 = me[1], me2 = me[2];

        // ∂h/∂δθ in the GLOBAL (world-frame) error convention (q_true = δq(δθ_world) ⊗ q̂):
        //   ∂(Rᵀ·m_earth)/∂δθ_world = Rᵀ·skew(m_earth).   FD- and loop-verified.
        // ∂h/∂m_earth = Rᵀ  (cols 9-11);  ∂h/∂m_body = I  (cols 12-14).
        // ∂h/∂k_I = currentAmps · I₃  (cols 16-18).
        // Rᵀ row i = [m[i], m[3+i], m[6+i]].
        const rowSkew = (r0, r1, r2) => [
            r1*me2 - r2*me1,
            -r0*me2 + r2*me0,
            r0*me1 - r1*me0,
        ];
        const t0 = rowSkew(m[0], m[3], m[6]);
        const t1 = rowSkew(m[1], m[4], m[7]);
        const t2 = rowSkew(m[2], m[5], m[8]);

        // State layout: δp(3) δv(3) δθ(3) b_a(3) b_g(3) m_earth(3) m_body(3) τ(1) k_I(3)
        cachedH = [
            [0,0,0, 0,0,0, t0[0],t0[1],t0[2], 0,0,0, 0,0,0, m[0],m[3],m[6], 1,0,0, 0, currentAmps,0,0],
            [0,0,0, 0,0,0, t1[0],t1[1],t1[2], 0,0,0, 0,0,0, m[1],m[4],m[7], 0,1,0, 0, 0,currentAmps,0],
            [0,0,0, 0,0,0, t2[0],t2[1],t2[2], 0,0,0, 0,0,0, m[2],m[5],m[8], 0,0,1, 0, 0,0,currentAmps],
        ];
        const hp = h(x);
        return [z[0]-hp[0], z[1]-hp[1], z[2]-hp[2]];
    }

    const defaultH = [
        [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,1,0,0, 0,currentAmps,0,0],
        [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,1,0, 0,0,currentAmps,0],
        [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, 0,0,0,0,0,1, 0,0,0,currentAmps],
    ];

    return {
        h,
        get H() { return cachedH || defaultH; },
        R: Rnoise,
        residual,
    };
}

/**
 * Declination pseudo-measurement.
 *
 * Soft constraint: atan2(magE, magN) ≈ WMM_declination (radians).
 * Keeps the earth field vector from drifting.
 *
 * @param {number} declRad - WMM declination at flight location (radians)
 * @param {number} [sigma=0.34] - noise in radians
 */
export function createDeclinationFactor(declRad, sigma = 0.34) {
    const varD = sigma * sigma;
    let cachedH = null;

    function h(x) {
        const me = x.mEarth || [0,0,0];
        return Math.atan2(me[1], me[0]);
    }

    function residual(z, x) {
        const me = x.mEarth || [0,0,0];
        const n2e2 = me[0]*me[0] + me[1]*me[1];
        const dHdN = n2e2 > 1e-12 ? -me[1]/n2e2 : 0;
        const dHdE = n2e2 > 1e-12 ? me[0]/n2e2 : 0;
        cachedH = [[0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0, dHdN,dHdE,0, 0,0,0]];
        return [z - h(x)];
    }

    return {
        h,
        get H() { return cachedH || [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]; },
        R: [[varD]],
        residual,
    };
}

/**
 * Map GPS satellite count to 1σ position noise in metres.
 *
 * @param {number} numSat  Number of satellites in view
 * @returns {number} 1σ position noise (m)
 */
export function computeGpsNoise(numSat) {
  if (numSat >= 12) return 1.5;
  if (numSat >= 8)  return 2.5;
  if (numSat >= 5)  return 4.0;
  return 8.0;
}

/**
 * GPS position measurement in NED with τ_gps latency.
 *
 * Measurement model: h = p − v·τ  (the GPS fix corresponds to state at t−τ).
 * ∂h/∂p = I₃,  ∂h/∂v = −τ·I₃,  ∂h/∂τ = −v.
 *
 * @param {{n: number, e: number, d: number}} meas  NED position (m)
 * @param {number} [sigma=2.5]  1σ noise in metres
 * @returns {object} factor
 */
export function createGpsPositionFactorWithLatency(meas, sigma = 2.5) {
    const varP = sigma * sigma;

    let cachedH = null;

    function h(x) {
        const tau = x.tauGps || 0;
        return [
            x.p[0] - x.v[0] * tau,
            x.p[1] - x.v[1] * tau,
            x.p[2] - x.v[2] * tau,
        ];
    }

    const R = [[varP, 0, 0], [0, varP, 0], [0, 0, varP]];

    function residual(z, x) {
        const tau = x.tauGps || 0;
        // H rows for 25-state: δp(0-2), δv(3-5), δθ(6-8), ba(9-11), bg(12-14), me(15-17), mb(18-20), τ(21), kI(22-24)
        cachedH = [
            [1,0,0, -tau,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, -x.v[0], 0,0,0],
            [0,1,0, 0,-tau,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, -x.v[1], 0,0,0],
            [0,0,1, 0,0,-tau, 0,0,0, 0,0,0, 0,0,0, 0,0,0, 0,0,0, -x.v[2], 0,0,0],
        ];
        const hp = h(x);
        return [z.n - hp[0], z.e - hp[1], z.d - hp[2]];
    }

    const defaultH = [
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    ];

    return {
        h,
        get H() { return cachedH || defaultH; },
        R,
        residual,
    };
}

export { GRAVITY_MAG };
