/**
 * IMU strapdown mechanization and SO(3) utilities for body-pose estimation.
 *
 * Conventions:
 *   Body frame:  FRD (Forward=X, Right=Y, Down=Z)
 *   World frame: NED (North, East, Down)
 *   Quaternions: Hamilton, body(FRD) → world(NED), scalar-first [w, x, y, z]
 *   Gravity:     [0, 0, +9.80665] (down is +Z in NED)
 *   Gyro:        rad/s in body FRD
 *   Accel:       m/s² in body FRD (includes gravity contribution)
 */

/** Gravitational acceleration in NED [m/s²] */
export const G_WORLD = Object.freeze([0, 0, 9.80665]);

// ---------------------------------------------------------------------------
// SO(3) utilities
// ---------------------------------------------------------------------------

/**
 * 3×3 skew-symmetric matrix from a 3-vector.
 * K = skew(v)  satisfies  K·w = v × w  for any 3-vector w.
 *
 * @param {number[]} v  3-vector [x, y, z]
 * @returns {number[][]}  3×3 skew-symmetric matrix
 */
export function skew(v) {
  const [x, y, z] = v;
  return [
    [0, -z, y],
    [z, 0, -x],
    [-y, x, 0],
  ];
}

/**
 * SO(3) exponential map (Rodrigues formula).
 * Maps a rotation vector φ (angle-axis) to a 3×3 rotation matrix R.
 *
 *   R = I + (sinθ / θ)·skew(φ) + ((1−cosθ) / θ²)·skew(φ)²
 *   θ = |φ|
 *
 * For θ → 0:  R ≈ I + skew(φ).
 *
 * @param {number[]} phi  Rotation vector [φx, φy, φz] in radians
 * @returns {number[][]}  3×3 rotation matrix
 */
export function expMap(phi) {
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

  const xy = x * y,
    xz = x * z,
    yz = y * z;
  const _bxy = b * xy,
    _bxz = b * xz,
    _byz = b * yz;

  // R = I + a·K + b·K²,  where K² = v·vᵀ − θ²·I
  return [
    [1 - b * (y * y + z * z), -a * z + _bxy, a * y + _bxz],
    [a * z + _bxy, 1 - b * (x * x + z * z), -a * x + _byz],
    [-a * y + _bxz, a * x + _byz, 1 - b * (x * x + y * y)],
  ];
}

/**
 * SO(3) logarithm map.
 * Computes the rotation vector φ from a 3×3 rotation matrix R such that
 * expMap(φ) = R.  The magnitude |φ| ∈ [0, π].
 *
 * For θ → 0 the result is obtained from the skew-symmetric part of R.
 * For θ ≈ π the rotation axis is extracted from R+I to avoid the
 * singularity of sinθ = 0.
 *
 * @param {number[][]} R  3×3 rotation matrix (RᵀR = I, det = 1)
 * @returns {number[]}  Rotation vector [φx, φy, φz] in radians
 */
export function logMap(R) {
  const r00 = R[0][0],
    r01 = R[0][1],
    r02 = R[0][2];
  const r10 = R[1][0],
    r11 = R[1][1],
    r12 = R[1][2];
  const r20 = R[2][0],
    r21 = R[2][1],
    r22 = R[2][2];

  const trace = r00 + r11 + r22;
  const cosT = Math.max(-1, Math.min(1, (trace - 1) * 0.5));

  if (cosT < -0.999999) {
    // Near π — the rotation axis comes from R+I (rank 1 at θ = π)
    let ux, uy, uz;
    if (r00 >= r11 && r00 >= r22) {
      ux = r00 + 1;
      uy = r10;
      uz = r20;
    } else if (r11 >= r22) {
      ux = r01;
      uy = r11 + 1;
      uz = r21;
    } else {
      ux = r02;
      uy = r12;
      uz = r22 + 1;
    }
    const len = Math.sqrt(ux * ux + uy * uy + uz * uz);
    if (len < 1e-12) return [0, 0, 0];
    const scl = Math.PI / len;
    return [scl * ux, scl * uy, scl * uz];
  }

  const theta = Math.acos(cosT);
  if (theta < 1e-12) {
    return [
      (r21 - r12) * 0.5,
      (r02 - r20) * 0.5,
      (r10 - r01) * 0.5,
    ];
  }

  const f = theta / (2 * Math.sin(theta));
  return [
    f * (r21 - r12),
    f * (r02 - r20),
    f * (r10 - r01),
  ];
}

/**
 * SO(3) right Jacobian Jr(φ).
 *
 * Relates the time-derivative of the exponential map to the tangent vector:
 *   Ṙ·Rᵀ = skew( Jr(φ) · φ̇ )
 *
 *   Jr(φ) = I − (1−cosθ)/θ² · K  +  (θ−sinθ)/θ³ · K²
 *   where  K = skew(φ),  θ = |φ|
 *
 * For θ → 0:  Jr ≈ I − ½·K + ⅙·K².
 *
 * @param {number[]} phi  Rotation vector [φx, φy, φz] in radians
 * @returns {number[][]}  3×3 right Jacobian matrix
 */
export function rightJacobian(phi) {
  const [x, y, z] = phi;
  const t2 = x * x + y * y + z * z;

  if (t2 < 1e-14) {
    // θ → 0: Jr ≈ I − ½·K + ⅙·K²
    const xx = x * x,
      yy = y * y,
      zz = z * z;
    const xy = x * y,
      xz = x * z,
      yz = y * z;
    return [
      [1 - (yy + zz) / 6, z * 0.5 + xy / 6, -y * 0.5 + xz / 6],
      [-z * 0.5 + xy / 6, 1 - (xx + zz) / 6, x * 0.5 + yz / 6],
      [y * 0.5 + xz / 6, -x * 0.5 + yz / 6, 1 - (xx + yy) / 6],
    ];
  }

  const theta = Math.sqrt(t2);
  const s = Math.sin(theta);
  const c = Math.cos(theta);
  const a = (1 - c) / t2;
  const b = (theta - s) / (theta * t2);

  const xx = x * x,
    yy = y * y,
    zz = z * z;
  const xy = x * y,
    xz = x * z,
    yz = y * z;

  // Jr = I − a·K + b·K²   (K² = v·vᵀ − θ²·I)
  return [
    [1 - b * (yy + zz), a * z + b * xy, -a * y + b * xz],
    [-a * z + b * xy, 1 - b * (xx + zz), a * x + b * yz],
    [a * y + b * xz, -a * x + b * yz, 1 - b * (xx + yy)],
  ];
}

// ---------------------------------------------------------------------------
// Quaternion utilities (Hamilton convention, scalar-first)
// ---------------------------------------------------------------------------

/**
 * Hamilton quaternion multiplication  q1 ⊗ q2.
 * q1 is applied after q2 (i.e. q1 ∘ q2 as rotations).
 *
 * @param {number[]} q1  Left quaternion [w, x, y, z]
 * @param {number[]} q2  Right quaternion [w, x, y, z]
 * @returns {number[]}  Product q1 ⊗ q2
 */
export function quatMultiply(q1, q2) {
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
 * Convert a Hamilton quaternion to a 3×3 rotation matrix (body → world).
 *
 *   v_world = R · v_body
 *
 * @param {number[]} q  Unit quaternion [w, x, y, z]
 * @returns {number[][]}  3×3 rotation matrix
 */
export function quatToRot(q) {
  const [w, x, y, z] = q;
  const xx = x * x,
    yy = y * y,
    zz = z * z;
  const xy = x * y,
    xz = x * z,
    yz = y * z;
  const wx = w * x,
    wy = w * y,
    wz = w * z;
  return [
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
  ];
}

/**
 * Convert a 3×3 rotation matrix to a unit Hamilton quaternion [w, x, y, z].
 * Uses Brent–Markley's numerically stable method that picks the largest trace
 * or diagonal element to avoid division by a near-zero quantity.
 *
 * @param {number[][]} R  3×3 rotation matrix
 * @returns {number[]}  Unit quaternion [w, x, y, z]
 */
export function rotToQuat(R) {
  const [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]] = R;
  const trace = r00 + r11 + r22;
  let q;

  if (trace > 0) {
    const s = 2 * Math.sqrt(trace + 1);
    q = [s * 0.25, (r21 - r12) / s, (r02 - r20) / s, (r10 - r01) / s];
  } else if (r00 > r11 && r00 > r22) {
    const s = 2 * Math.sqrt(1 + r00 - r11 - r22);
    q = [(r21 - r12) / s, s * 0.25, (r01 + r10) / s, (r02 + r20) / s];
  } else if (r11 > r22) {
    const s = 2 * Math.sqrt(1 + r11 - r00 - r22);
    q = [(r02 - r20) / s, (r01 + r10) / s, s * 0.25, (r12 + r21) / s];
  } else {
    const s = 2 * Math.sqrt(1 + r22 - r00 - r11);
    q = [(r10 - r01) / s, (r02 + r20) / s, (r12 + r21) / s, s * 0.25];
  }

  const n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/**
 * Build a unit quaternion from an axis-angle representation.
 * The axis is normalised internally.
 *
 * @param {number[]} axis   Rotation axis [ax, ay, az] (need not be unit)
 * @param {number}   angle  Rotation angle in radians (right-hand rule)
 * @returns {number[]}  Unit quaternion [w, x, y, z]
 */
export function quatFromAxisAngle(axis, angle) {
  const [ax, ay, az] = axis;
  const n = Math.sqrt(ax * ax + ay * ay + az * az);
  if (n < 1e-14) {
    return [1, 0, 0, 0];
  }
  const half = angle * 0.5;
  const s = Math.sin(half) / n;
  return [Math.cos(half), s * ax, s * ay, s * az];
}

/**
 * Extract ZYX intrinsic Euler angles from a quaternion.
 * The quaternion maps body(FRD) → world(NED).
 *
 * Order of rotations (applied first-to-last): yaw (Z), pitch (Y), roll (X).
 *
 * @param {number[]} q  Unit quaternion [w, x, y, z]
 * @returns {{roll: number, pitch: number, yaw: number}}  Angles in radians
 */
export function quatToEuler(q) {
  const R = quatToRot(q);
  const roll = Math.atan2(R[2][1], R[2][2]);
  const pitch = -Math.asin(Math.max(-1, Math.min(1, R[2][0])));
  const yaw = Math.atan2(R[1][0], R[0][0]);
  return { roll, pitch, yaw };
}

/**
 * Build a unit quaternion from ZYX intrinsic Euler angles (radians).
 * The resulting quaternion maps body(FRD) → world(NED) with standard
 * right-handed yaw (0°=North, CW+).
 *
 * Order of rotations (applied first-to-last): yaw about Z, pitch about Y, roll about X.
 *
 * @param {number} roll  - radians
 * @param {number} pitch - radians
 * @param {number} yaw   - radians
 * @returns {number[]} Unit quaternion [w, x, y, z]
 */
export function eulerToQuat(roll, pitch, yaw) {
    const cr = Math.cos(roll * 0.5), sr = Math.sin(roll * 0.5);
    const cp = Math.cos(pitch * 0.5), sp = Math.sin(pitch * 0.5);
    const cy = Math.cos(yaw * 0.5), sy = Math.sin(yaw * 0.5);
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ];
}

// ---------------------------------------------------------------------------
// IMU strapdown mechanization
// ---------------------------------------------------------------------------

/**
 * One step of continuous strapdown inertial navigation (explicit Euler).
 *
 * The raw accelerometer reading includes gravity with the MEMS convention:
 * a level drone at rest reads [+g] on the body +Z axis (down in FRD).
 * The strapdown equation uses specific force (a_body − g_body), which is
 * the negative of the sensor reading.
 *
 *   f_body = −(accel − b_a)
 *   q_new = normalize( q + ½·q ⊗ (ω − b_g)·dt )
 *   v_new = v + R(q)·f_body·dt + g_world·dt
 *   p_new = p + v·dt + ½·R(q)·f_body·dt² + ½·g_world·dt²
 *
 * @param {number[]} omega  Gyroscope reading [ωx, ωy, ωz] in body FRD (rad/s)
 * @param {number[]} accel  Raw accelerometer reading [ax, ay, az] in body FRD (m/s², includes gravity, reads +g on +Z at rest)
 * @param {number[]} q      Current quaternion [w, x, y, z] (scalar-first)
 * @param {number[]} v      Current velocity in world NED [vn, ve, vd] (m/s)
 * @param {number[]} p      Current position in world NED [n, e, d] (m)
 * @param {number}   dt     Time step (s)
 * @param {number[]} [b_g]  Gyroscope bias [3] rad/s (default zeros)
 * @param {number[]} [b_a]  Accelerometer bias [3] m/s² (default zeros)
 * @returns {{q: number[], v: number[], p: number[]}}  New state
 */
export function strapdownPropagate(omega, accel, q, v, p, dt, b_g, b_a) {
    const bg = b_g || [0, 0, 0];
    const ba = b_a || [0, 0, 0];

    const wx = omega[0] - bg[0];
    const wy = omega[1] - bg[1];
    const wz = omega[2] - bg[2];

    // Specific force = negative of de-biased accelerometer reading
    const fx = -(accel[0] - ba[0]);
    const fy = -(accel[1] - ba[1]);
    const fz = -(accel[2] - ba[2]);

  // -- Orientation: q̇ = ½ q ⊗ [0, ω_corr] --------------------------------
  const qw = q[0],
    qx = q[1],
    qy = q[2],
    qz = q[3];
  const hdt = dt * 0.5;

  let qwn = qw + hdt * (-qx * wx - qy * wy - qz * wz);
  let qxn = qx + hdt * (qw * wx + qy * wz - qz * wy);
  let qyn = qy + hdt * (-qx * wz + qw * wy + qz * wx);
  let qzn = qz + hdt * (qx * wy - qy * wx + qw * wz);

  const qn =
    Math.sqrt(qwn * qwn + qxn * qxn + qyn * qyn + qzn * qzn);
  qwn /= qn;
  qxn /= qn;
  qyn /= qn;
  qzn /= qn;

  // -- Rotate body acceleration to world frame  R(q)·a_corr --------------
  const R00 = 1 - 2 * (qy * qy + qz * qz);
  const R01 = 2 * (qx * qy - qw * qz);
  const R02 = 2 * (qx * qz + qw * qy);
  const R10 = 2 * (qx * qy + qw * qz);
  const R11 = 1 - 2 * (qx * qx + qz * qz);
  const R12 = 2 * (qy * qz - qw * qx);
  const R20 = 2 * (qx * qz - qw * qy);
  const R21 = 2 * (qy * qz + qw * qx);
  const R22 = 1 - 2 * (qx * qx + qy * qy);

  const awx = R00 * fx + R01 * fy + R02 * fz;
  const awy = R10 * fx + R11 * fy + R12 * fz;
  const awz = R20 * fx + R21 * fy + R22 * fz;

  const hdt2 = 0.5 * dt * dt;

  return {
    q: [qwn, qxn, qyn, qzn],
    v: [
      v[0] + awx * dt,
      v[1] + awy * dt,
      v[2] + (awz + 9.80665) * dt,
    ],
    p: [
      p[0] + v[0] * dt + awx * hdt2,
      p[1] + v[1] * dt + awy * hdt2,
      p[2] + v[2] * dt + (awz + 9.80665) * hdt2,
    ],
  };
}
