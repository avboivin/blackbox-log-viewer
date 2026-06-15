// Rauch-Tung-Striebel (RTS) fixed-interval smoother for the ESKF error state.
// State vector (Q3: unconditional 25-state): [δp(3), δv(3), δθ(3), b_a(3), b_g(3), δm_earth(3), δm_body(3), δτ(1), δk_I(3)].
// The backward pass distributes measurement corrections over the preceding
// IMU-propagated sub-trajectory in closed form.

import { quatFromAxisAngle } from "./imuMechanization.js";

// ---------------------------------------------------------------------------
// Matrix helpers  (n × n dense matrices stored as array-of-arrays)
// ---------------------------------------------------------------------------

/**
 * Transpose of a square matrix.
 */
function matrixTranspose(M) {
  const n = M.length;
  const T = new Array(n);
  for (let i = 0; i < n; i++) {
    T[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      T[i][j] = M[j][i];
    }
  }
  return T;
}

/**
 * Square matrix multiplication  C = A · B   (all n×n).
 */
function matrixMultiply(A, B) {
  const n = A.length;
  const C = new Array(n);
  for (let i = 0; i < n; i++) {
    C[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += A[i][k] * B[k][j];
      }
      C[i][j] = sum;
    }
  }
  return C;
}

/**
 * Matrix-vector multiply  y = M · v   (M is n×n, v is length-n, returns length-n).
 */
function matrixVectorMultiply(M, v) {
  const n = M.length;
  const y = new Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < n; j++) {
      sum += M[i][j] * v[j];
    }
    y[i] = sum;
  }
  return y;
}

/**
 * Element-wise matrix addition  A + B.
 */
function matrixAdd(A, B) {
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

/**
 * Element-wise matrix subtraction  A − B.
 */
function matrixSub(A, B) {
  const n = A.length;
  const C = new Array(n);
  for (let i = 0; i < n; i++) {
    C[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      C[i][j] = A[i][j] - B[i][j];
    }
  }
  return C;
}

/**
 * Gauss-Jordan matrix inversion with partial pivoting.
 * Returns a new n×n matrix.
 *
 * @throws {Error} if the matrix is singular (no valid pivot found).
 */
function matrixInverse(M) {
  const n = M.length;
  // Augmented matrix [M | I]
  const aug = new Array(n);
  for (let i = 0; i < n; i++) {
    aug[i] = new Array(2 * n);
    for (let j = 0; j < n; j++) {
      aug[i][j] = M[i][j];
      aug[i][j + n] = i === j ? 1 : 0;
    }
  }

  for (let col = 0; col < n; col++) {
    // Partial pivoting: find largest absolute value in column col below the diagonal
    let maxRow = col;
    let maxVal = Math.abs(aug[col][col]);
    for (let i = col + 1; i < n; i++) {
      const absVal = Math.abs(aug[i][col]);
      if (absVal > maxVal) {
        maxVal = absVal;
        maxRow = i;
      }
    }
    if (maxVal < 1e-14) {
      throw new Error("Matrix is singular");
    }
    if (maxRow !== col) {
      const tmp = aug[col];
      aug[col] = aug[maxRow];
      aug[maxRow] = tmp;
    }

    // Scale pivot row
    const pivot = aug[col][col];
    for (let j = col; j < 2 * n; j++) {
      aug[col][j] /= pivot;
    }

    // Eliminate all other rows
    for (let i = 0; i < n; i++) {
      if (i === col) continue;
      const factor = aug[i][col];
      for (let j = col; j < 2 * n; j++) {
        aug[i][j] -= factor * aug[col][j];
      }
    }
  }

  const inv = new Array(n);
  for (let i = 0; i < n; i++) {
    inv[i] = aug[i].slice(n);
  }
  return inv;
}

// ---------------------------------------------------------------------------
// Quaternion helpers  (quaternion stored as [w, x, y, z])
// ---------------------------------------------------------------------------

/**
 * Hamilton quaternion multiplication  q1 ⊗ q2.
 * q1 is applied after q2 as a rotation.
 */
function quatMultiply(q1, q2) {
  const [w1, x1, y1, z1] = q1;
  const [w2, x2, y2, z2] = q2;
  return [
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ];
}

/**
 * Normalise a quaternion to unit length.
 */
function quatNormalize(q) {
  const [w, x, y, z] = q;
  const n = Math.sqrt(w * w + x * x + y * y + z * z);
  if (n < 1e-14) {
    return [1, 0, 0, 0];
  }
  return [w / n, x / n, y / n, z / n];
}

/**
 * Quaternion conjugate (inverse for unit quaternion).
 */
function quatConjugate(q) {
  return [q[0], -q[1], -q[2], -q[3]];
}

/**
 * Convert a unit quaternion to its equivalent rotation vector.
 * The rotation vector is ω = θ · u  where u is the unit axis and θ is the angle.
 *
 * Uses atan2 for numerical stability near small angles.
 */
function quatToRotationVector(q) {
  const w = q[0];
  const vx = q[1];
  const vy = q[2];
  const vz = q[3];
  const vNorm = Math.sqrt(vx * vx + vy * vy + vz * vz);
  if (vNorm < 1e-14) {
    return [0, 0, 0];
  }
  const theta = 2 * Math.atan2(vNorm, w);
  const scale = theta / vNorm;
  return [vx * scale, vy * scale, vz * scale];
}

// ---------------------------------------------------------------------------
// State vector helpers  (error state δx ∈ R⁹)
// ---------------------------------------------------------------------------

/**
 * Deep-copy a nominal state  { p, v, q, mEarth?, mBody?, tauGps?, kI?, tUs? }.
 */
function copyState(x) {
  const s = {
    p: [...x.p],
    v: [...x.v],
    q: [...x.q],
  };
  if (x.ba !== undefined) s.ba = [...x.ba];
  if (x.bg !== undefined) s.bg = [...x.bg];
  if (x.mEarth !== undefined) s.mEarth = [...x.mEarth];
  if (x.mBody !== undefined) s.mBody = [...x.mBody];
  if (x.tauGps !== undefined) s.tauGps = x.tauGps;
  if (x.kI !== undefined) s.kI = [...x.kI];
  if (x.tUs !== undefined) s.tUs = x.tUs;
  return s;
}

/**
 * Deep-copy an n×n matrix.
 */
function copyMatrix(M) {
  return M.map((row) => [...row]);
}

/**
 * Compute the error-state difference (x_a ⊖ x_b) as an n-vector.
 *
 * Produces a vector with fixed state layout matching the effective dimension.
 * Fields absent on both sides produce zero at their index.
 *
 * State indices (Q3: unconditional 25-state):
 *   0–2: δp, 3–5: δv, 6–8: δθ,
 *   9–11: b_a, 12–14: b_g,
 *   15–17: mEarth, 18–20: mBody, 21: τ_gps, 22–24: k_I
 *
 * @param {{p: number[], v: number[], q: number[], ba?: number[], bg?: number[], mEarth?: number[], mBody?: number[]}} xSmooth
 * @param {{p: number[], v: number[], q: number[], ba?: number[], bg?: number[], mEarth?: number[], mBody?: number[]}} xPred
 * @returns {number[]} n-element error state
 */
function stateDifference(xSmooth, xPred) {
  const dp = [
    xSmooth.p[0] - xPred.p[0],
    xSmooth.p[1] - xPred.p[1],
    xSmooth.p[2] - xPred.p[2],
  ];
  const dv = [
    xSmooth.v[0] - xPred.v[0],
    xSmooth.v[1] - xPred.v[1],
    xSmooth.v[2] - xPred.v[2],
  ];
  const qErr = quatMultiply(xSmooth.q, quatConjugate(xPred.q));
  const dTheta = quatToRotationVector(quatNormalize(qErr));

  // Unconditional bias states (Q3): indices 9-14
  const dBa = (xSmooth.ba && xPred.ba) ? [
    xSmooth.ba[0] - xPred.ba[0],
    xSmooth.ba[1] - xPred.ba[1],
    xSmooth.ba[2] - xPred.ba[2],
  ] : [0, 0, 0];
  const dBg = (xSmooth.bg && xPred.bg) ? [
    xSmooth.bg[0] - xPred.bg[0],
    xSmooth.bg[1] - xPred.bg[1],
    xSmooth.bg[2] - xPred.bg[2],
  ] : [0, 0, 0];

  const result = [...dp, ...dv, ...dTheta, ...dBa, ...dBg];

  function has(field) {
    return (xSmooth[field] !== undefined) || (xPred[field] !== undefined);
  }

  // mEarth: indices 15–17
  if (xSmooth.mEarth !== undefined && xPred.mEarth !== undefined) {
    result.push(xSmooth.mEarth[0] - xPred.mEarth[0], xSmooth.mEarth[1] - xPred.mEarth[1], xSmooth.mEarth[2] - xPred.mEarth[2]);
  } else if (has("mEarth") || has("mBody") || has("tauGps") || has("kI")) {
    result.push(0, 0, 0);
  }
  // mBody: indices 18–20
  if (xSmooth.mBody !== undefined && xPred.mBody !== undefined) {
    result.push(xSmooth.mBody[0] - xPred.mBody[0], xSmooth.mBody[1] - xPred.mBody[1], xSmooth.mBody[2] - xPred.mBody[2]);
  } else if (has("mBody") || has("tauGps") || has("kI")) {
    result.push(0, 0, 0);
  }
  // τ_gps: index 21
  if (xSmooth.tauGps !== undefined && xPred.tauGps !== undefined) {
    result.push(xSmooth.tauGps - xPred.tauGps);
  } else if (has("tauGps") || has("kI")) {
    result.push(0);
  }
  // k_I: indices 22–24
  if (xSmooth.kI !== undefined && xPred.kI !== undefined) {
    result.push(xSmooth.kI[0] - xPred.kI[0], xSmooth.kI[1] - xPred.kI[1], xSmooth.kI[2] - xPred.kI[2]);
  } else if (has("kI")) {
    result.push(0, 0, 0);
  }
  return result;
}

/**
 * Apply an error-state correction (⊕) to a nominal state.
 *
 *   p' = p + δp
 *   v' = v + δv
 *   q' = dq ⊗ q   where dq = quatFromAxisAngle(δθ_axis, |δθ|)
 *   ba' = ba + δba   (indices 9–11; unconditional Q3)
 *   bg' = bg + δbg   (indices 12–14; unconditional Q3)
 *   mEarth' = mEarth + δm_earth   (indices 15–17)
 *   mBody'  = mBody  + δm_body    (indices 18–20)
 *   τ_gps' = τ_gps + δτ           (index 21)
 *   k_I'   = k_I + δk_I           (indices 22–24)
 *
 * @param {{p: number[], v: number[], q: number[], ba?: number[], bg?: number[], mEarth?: number[], mBody?: number[]}} x
 * @param {number[]} deltaX  n-element error state
 * @returns {{p: number[], v: number[], q: number[], ba?: number[], bg?: number[], mEarth?: number[], mBody?: number[]}}
 */
function stateAdd(x, deltaX) {
  const p = [x.p[0] + deltaX[0], x.p[1] + deltaX[1], x.p[2] + deltaX[2]];
  const v = [x.v[0] + deltaX[3], x.v[1] + deltaX[4], x.v[2] + deltaX[5]];

  const dThetaX = deltaX[6];
  const dThetaY = deltaX[7];
  const dThetaZ = deltaX[8];
  const thetaNorm = Math.sqrt(
    dThetaX * dThetaX + dThetaY * dThetaY + dThetaZ * dThetaZ,
  );
  let q;
  if (thetaNorm < 1e-14) {
    q = [...x.q];
  } else {
    const axis = [
      dThetaX / thetaNorm,
      dThetaY / thetaNorm,
      dThetaZ / thetaNorm,
    ];
    const dq = quatFromAxisAngle(axis, thetaNorm);
    q = quatNormalize(quatMultiply(dq, x.q));
  }
  const result = { p, v, q };
  if (x.tUs !== undefined) result.tUs = x.tUs;

  // Unconditional bias states (Q3): indices 9–14
  if (deltaX.length >= 12 && x.ba !== undefined) {
    result.ba = [x.ba[0] + deltaX[9], x.ba[1] + deltaX[10], x.ba[2] + deltaX[11]];
  }
  if (deltaX.length >= 15 && x.bg !== undefined) {
    result.bg = [x.bg[0] + deltaX[12], x.bg[1] + deltaX[13], x.bg[2] + deltaX[14]];
  }
  // Mag + nuisance states: indices 15–24
  if (deltaX.length >= 18 && x.mEarth !== undefined) {
    result.mEarth = [x.mEarth[0] + deltaX[15], x.mEarth[1] + deltaX[16], x.mEarth[2] + deltaX[17]];
  }
  if (deltaX.length >= 21 && x.mBody !== undefined) {
    result.mBody = [x.mBody[0] + deltaX[18], x.mBody[1] + deltaX[19], x.mBody[2] + deltaX[20]];
  }
  if (deltaX.length >= 22 && x.tauGps !== undefined) {
    result.tauGps = x.tauGps + deltaX[21];
  }
  if (deltaX.length >= 25 && x.kI !== undefined) {
    result.kI = [x.kI[0] + deltaX[22], x.kI[1] + deltaX[23], x.kI[2] + deltaX[24]];
  }
  return result;
}

// ---------------------------------------------------------------------------
// RTS fixed-interval smoother
// ---------------------------------------------------------------------------

/**
 * Apply the Rauch-Tung-Striebel backward smoothing pass to an error-state
 * Kalman filter trajectory.
 *
 * The smoother uses the error-state transition matrices F_k (step k → k+1)
 * to optimally distribute measurement corrections backward over the IMU-
 * propagated sub-trajectory.
 *
 * For each step k from N−1 down to 0:
 *
 *   C_k  =  P_{k|k} · F_k^T · inv(P_{k+1|k})
 *   x_{k|N}  =  x_{k|k}  ⊕  C_k · (x_{k+1|N} ⊖ x_{k+1|k})
 *   P_{k|N}  =  P_{k|k}  +  C_k · (P_{k+1|N} − P_{k+1|k}) · C_k^T
 *
 * If F_k is null/undefined the filter step is copied through unchanged.
 *
 * @param {Array<{x: {p: number[], v: number[], q: number[], mEarth?: number[], mBody?: number[]}, P: number[][], xPred: {p: number[], v: number[], q: number[], mEarth?: number[], mBody?: number[]}|null, PPred: number[][]|null}>} filterResults
 *   Forward-filter estimates, length N+1 (indices 0 … N).
 *   xPred / PPred may be null only at k=0.
 * @param {Array<number[][]|null>} transitionMatrices
 *   Error-state transition matrices F_k, length N (one fewer than filterResults).
 * @returns {Array<{x: {p: number[], v: number[], q: number[], mEarth?: number[], mBody?: number[]}, P: number[][]}>}
 *   Smoothed estimates x_{k|N}, P_{k|N} for every timestep.
 */
export function rtsSmooth(filterResults, transitionMatrices) {
  const Np1 = filterResults.length;
  if (Np1 === 0) {
    return [];
  }

  // Allocate and initialise the last step from the forward-pass filtered estimate.
  const smoothed = new Array(Np1);
  smoothed[Np1 - 1] = {
    x: copyState(filterResults[Np1 - 1].x),
    P: copyMatrix(filterResults[Np1 - 1].P),
  };

  // Backward pass: k from N−2 down to 0
  for (let k = Np1 - 2; k >= 0; k--) {
    const Fk = transitionMatrices[k];

    if (!Fk) {
      // No valid transition — copy filtered directly.
      smoothed[k] = {
        x: copyState(filterResults[k].x),
        P: copyMatrix(filterResults[k].P),
      };
      continue;
    }

    const Pk = filterResults[k].P;
    // Predicted covariance at step k+1 (before update).  Fall back to
    // filtered covariance if PPred is missing (e.g. first step or
    // measurementless step where P == PPred).
    const Pkp1Pred = filterResults[k + 1].PPred || filterResults[k + 1].P;
    const Pkp1Smoothed = smoothed[k + 1].P;

    // Nominal states
    // xPred at step k+1 is needed for the state difference.  If missing,
    // fall back to the filtered state (identity error).
    const xkp1Pred = filterResults[k + 1].xPred || filterResults[k + 1].x;
    const xkp1Smoothed = smoothed[k + 1].x;

    let Ck;
    try {
      const FkT = matrixTranspose(Fk);
      const PFt = matrixMultiply(Pk, FkT);
      const invPred = matrixInverse(Pkp1Pred);
      Ck = matrixMultiply(PFt, invPred);
    } catch {
      // Inversion failed — identity smoother step.
      smoothed[k] = {
        x: copyState(filterResults[k].x),
        P: copyMatrix(filterResults[k].P),
      };
      continue;
    }

    // State update: δx = C_k · (x_{k+1|N} ⊖ x_{k+1|k})
    const deltaNext = stateDifference(xkp1Smoothed, xkp1Pred);
    const deltaX = matrixVectorMultiply(Ck, deltaNext);
    smoothed[k] = {
      x: stateAdd(filterResults[k].x, deltaX),
      P: (function () {
        // P_{k|N} = P_{k|k} + C_k · (P_{k+1|N} − P_{k+1|k}) · C_k^T
        const dP = matrixSub(Pkp1Smoothed, Pkp1Pred);
        const CdP = matrixMultiply(Ck, dP);
        const CdPCt = matrixMultiply(CdP, matrixTranspose(Ck));
        return matrixAdd(Pk, CdPCt);
      })(),
    };
  }

  return smoothed;
}
