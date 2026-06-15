/**
 * Synthetic trajectory generator and sensor simulator.
 *
 * Plants a known ground-truth trajectory then simulates the sensor logs
 * (GPS, IMU, baro, mag) that Betaflight would produce for that motion.
 * Used with a fixed PRNG seed to create deterministic test fixtures for the
 * estimator — the estimator must recover the planted pose within the injected
 * noise budget.
 */

/**
 * Simple deterministic PRNG (mulberry32).
 * @param {number} seed
 * @returns {function(): number} generator returning numbers in [0, 1)
 */
export function createRng(seed) {
    let s = seed | 0;
    return function () {
        s = (s + 0x6d2b79f5) | 0;
        let t = Math.imul(s ^ (s >>> 15), 1 | s);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * Normal distribution via Box-Muller.
 * @param {function(): number} rng
 * @param {number} [mean=0]
 * @param {number} [std=1]
 * @returns {number}
 */
export function randn(rng, mean = 0, std = 1) {
    const u1 = rng() || 1e-9;
    const u2 = rng();
    return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/**
 * Build a synthetic trajectory: a circular path with constant speed and
 * bank angle, plus a straight section.
 *
 * @param {object} opts
 * @param {number} [opts.durationS=10] - flight duration in seconds
 * @param {number} [opts.freqHz=100] - sample rate for the ground truth
 * @param {number} [opts.radiusM=50] - circle radius
 * @param {number} [opts.speedMs=15] - forward speed
 * @param {number} [opts.originLat=48.408] - NED origin latitude
 * @param {number} [opts.originLon=-71.164] - NED origin longitude
 * @param {number} [opts.originAlt=200] - NED origin altitude MSL
 * @returns {Array<SyntheticPose>}
 */
export function generateCircularTrajectory(opts = {}) {
    const {
        durationS = 10,
        freqHz = 100,
        radiusM = 50,
        speedMs = 15,
        originAlt = 200,
    } = opts;

    const dt = 1 / freqHz;
    const N = Math.floor(durationS * freqHz);
    const poses = [];

    const angularRate = speedMs / radiusM;

    for (let i = 0; i < N; i++) {
        const t = i * dt;
        const heading = angularRate * t;
        const tn = heading;

        // Position in NED: circle in the NE plane at constant height
        const pN = radiusM * Math.sin(tn);
        const pE = radiusM * (1 - Math.cos(tn));
        const pD = -originAlt;

        // Velocity in NED: tangent to the circle
        const vN = speedMs * Math.cos(tn);
        const vE = speedMs * Math.sin(tn);
        const vD = 0;

        // Attitude: yaw rotation only for horizontal circle
        const qSimple = [
            Math.cos(heading / 2),
            0,
            0,
            Math.sin(heading / 2),
        ];

        poses.push({
            t,
            pNed: { n: pN, e: pE, d: pD },
            vNed: { n: vN, e: vE, d: vD },
            q: qSimple,
            heading,
            bankAngle: 0,
        });
    }

    return poses;
}

/**
 * Generate a straight-and-level trajectory.
 *
 * @param {object} opts
 * @param {number} [opts.durationS=5]
 * @param {number} [opts.speedMs=10]
 * @param {number} [opts.headingDeg=45]
 * @returns {Array<SyntheticPose>}
 */
export function generateStraightTrajectory(opts = {}) {
    const {
        durationS = 5,
        speedMs = 10,
        headingDeg = 45,
        freqHz = 100,
    } = opts;

    const heading = (headingDeg * Math.PI) / 180;
    const dt = 1 / freqHz;
    const N = Math.floor(durationS * freqHz);
    const poses = [];

    for (let i = 0; i < N; i++) {
        const t = i * dt;
        poses.push({
            t,
            pNed: {
                n: speedMs * Math.cos(heading) * t,
                e: speedMs * Math.sin(heading) * t,
                d: -200,
            },
            vNed: {
                n: speedMs * Math.cos(heading),
                e: speedMs * Math.sin(heading),
                d: 0,
            },
            q: [
                Math.cos(heading / 2),
                0,
                0,
                Math.sin(heading / 2),
            ],
            heading,
            bankAngle: 0,
        });
    }

    return poses;
}

/**
 * @typedef {object} SyntheticPose
 * @property {number} t - time in seconds
 * @property {{n:number,e:number,d:number}} pNed
 * @property {{n:number,e:number,d:number}} vNed
 * @property {[number,number,number,number]} q - [w,x,y,z] body(FRD)→world(NED)
 * @property {number} heading - yaw in radians
 * @property {number} bankAngle - roll in radians
 */

// ---------------------------------------------------------------------------
// Dynamic trajectory generator — non-level, rotating attitudes
// ---------------------------------------------------------------------------

/**
 * Quaternion from ZYX Euler angles (radians), scalar-first Hamilton.
 */
function eulerToQuat(roll, pitch, yaw) {
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

/**
 * Quaternion to 3×3 rotation matrix (body FRD → world NED).
 */
function quatToRotMatrix(q) {
    const [w, x, y, z] = q;
    const xx = x * x, yy = y * y, zz = z * z;
    const xy = x * y, xz = x * z, yz = y * z;
    const wx = w * x, wy = w * y, wz = w * z;
    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ];
}


/**
 * Generate a trajectory with dynamic attitudes: banked turn → climb → yaw sweep.
 *
 * Segment 1 (0–T1): Coordinated right banking turn at constant speed and altitude.
 *   Roll = −bankAngle (right bank is negative roll about body X-forward).
 *   Yaw rate = v / r, turn radius r = v²/(g·tan|φ|).
 *
 * Segment 2 (T1–T2): Climbing straight, pitch up.
 *
 * Segment 3 (T2–T3): Flat yaw sweep (hovering spin).
 *
 * @param {object} [opts]
 * @param {number} [opts.freqHz=200] - internal sample rate
 * @returns {{ traj: SyntheticPose[], params: object }}
 */
export function generateDynamicTrajectory(opts = {}) {
    const { freqHz = 200 } = opts;
    const dt = 1 / freqHz;
    const g = 9.80665;

    // Segment 1: banked turn
    const T1 = 2.0;
    const speed = 15;
    const bankDeg = 30;
    const bank = (bankDeg * Math.PI) / 180;
    const radius = (speed * speed) / (g * Math.tan(Math.abs(bank)));
    const yawRate = speed / radius;
    const N1 = Math.floor(T1 * freqHz);

    // Segment 2: climbing
    const T2 = T1 + 2.0;
    const climbPitchDeg = 15;
    const climbPitch = (climbPitchDeg * Math.PI) / 180;
        const N2 = Math.floor((T2 - T1) * freqHz);

    // Segment 3: yaw sweep
    const T3 = T2 + 1.5;
    const spinRate = Math.PI;  // π rad/s
    const N3 = Math.floor((T3 - T2) * freqHz);

    const poses = [];

    // Integration state — initialised inside segment loops
    let pN = 0, pE = 0, pD = -200;
    let vN, vE, vD;
    let roll, pitch, yaw = 0;

    // Segment 1: banked right turn
    for (let i = 0; i < N1; i++) {
        const t = i * dt;
        yaw += yawRate * dt;
        roll = -bank;
        pitch = 0;
        vD = 0;

        const q = eulerToQuat(roll, pitch, yaw);
        // Velocity in world frame
        vN = speed * Math.cos(yaw);
        vE = speed * Math.sin(yaw);

        pN = radius * Math.sin(yaw);
        pE = radius * (1 - Math.cos(yaw));

        poses.push({
            t, pNed: { n: pN, e: pE, d: pD },
            vNed: { n: vN, e: vE, d: vD },
            q, heading: yaw, bankAngle: roll, pitch,
        });
    }

    // Segment 2: climbing straight
    for (let i = 0; i < N2; i++) {
        const j = N1 + i;
        const t = j * dt;
        
        roll = -bank;
        pitch = climbPitch;

        const q = eulerToQuat(roll, pitch, yaw);
        vN = speed * Math.cos(yaw) * Math.cos(climbPitch);
        vE = speed * Math.sin(yaw) * Math.cos(climbPitch);
        vD = -speed * Math.sin(climbPitch);

        pN += vN * dt;
        pE += vE * dt;
        pD += vD * dt;

        poses.push({
            t, pNed: { n: pN, e: pE, d: pD },
            vNed: { n: vN, e: vE, d: vD },
            q, heading: yaw, bankAngle: roll, pitch,
        });
    }

    // Segment 3: flat yaw sweep (hover)
    for (let i = 0; i < N3; i++) {
        const j = N1 + N2 + i;
        const t = j * dt;
        
        yaw += spinRate * dt;
        roll = 0;
        pitch = 0;

        const q = eulerToQuat(roll, pitch, yaw);

        poses.push({
            t, pNed: { n: pN, e: pE, d: pD },
            vNed: { n: 0, e: 0, d: 0 },
            q, heading: yaw, bankAngle: roll, pitch,
        });
    }

    return {
        traj: poses,
        params: { T1, T2, T3, speed, bankDeg, climbPitchDeg, radius, freqHz },
    };
}

// ---------------------------------------------------------------------------
// Sensor stream synthesizer
// ---------------------------------------------------------------------------

/**
 * Synthesize sensor streams from a ground-truth trajectory.
 *
 * The accelerometer raw reading = −(specific force), so a level drone at rest
 * reads [+g] on body +Z (FRD down). Gyro = angular velocity in body FRD.
 * Mag body = Rᵀ · m_earth_world + m_body (in Gauss).
 *
 * @param {SyntheticPose[]} traj
 * @param {object} [opts]
 * @param {number[]} [opts.mEarth=[0.17,-0.047,0.51]] - earth field NED (Gauss)
 * @param {number[]} [opts.mBody=[0,0,0]] - body hard iron (Gauss)
 * @param {function} [opts.rng] - PRNG for noise
 * @param {number} [opts.gpsNoiseStd=0] - GPS position noise 1σ (m), 0 = noise-free
 * @param {number} [opts.gyroNoiseStd=0] - gyro noise 1σ (rad/s)
 * @param {number} [opts.accelNoiseStd=0] - accel noise 1σ (m/s²)
 * @param {{lat:number, lon:number, alt:number}} [opts.origin] - geodetic origin for GPS alt round-trip
 * @returns {{ imu: ImuSample[], gps: GpsFix[], baro: BaroSample[], quat: object[], mag: object[] }}
 */
export function generateSensorStreams(traj, opts = {}) {
    const {
        mEarth = [0.17, -0.047, 0.51],
        mBody = [0, 0, 0],
        rng = null,
        gpsNoiseStd: _gpsNoiseStd = 0,
        gyroNoiseStd = 0,
        accelNoiseStd = 0,
        origin = null,
    } = opts;

    const g = 9.80665;
    const originAlt = origin ? origin.alt : 0;
    const imu = [], gpsList = [], baro = [], quat = [], mag = [];
    let lastQ = null;
    const dt = traj.length > 1 ? traj[1].t - traj[0].t : 0.01;

    for (let i = 0; i < traj.length; i++) {
        const pose = traj[i];
        const tUs = Math.round(pose.t * 1e6);

        // ---- Gyroscope: angular velocity in body FRD ----
        let gyro = [0, 0, 0];
        if (lastQ && dt > 0) {
            // q = q_last ⊗ exp(½·ω·dt)  →  ω = 2·log(q_last⁻¹ ⊗ q) / dt
            const qLastConj = [lastQ[0], -lastQ[1], -lastQ[2], -lastQ[3]];
            const qRel = quatMult(qLastConj, pose.q);
            const rotVec = quatToRotVec(qRel);
            gyro = [
                rotVec[0] / dt + (rng ? randn(rng, 0, gyroNoiseStd) : 0),
                rotVec[1] / dt + (rng ? randn(rng, 0, gyroNoiseStd) : 0),
                rotVec[2] / dt + (rng ? randn(rng, 0, gyroNoiseStd) : 0),
            ];
        }
        lastQ = pose.q;

        // ---- Accelerometer: raw = −specific force = −Rᵀ·(a_world − g) ----
        // For a drone, a_world is the true kinematic acceleration.
        // Numerically differentiate velocity for a_world.
        let aWorld = [0, 0, 0];
        if (i >= 2) {
            const dtA = traj[i].t - traj[i - 1].t;
            if (dtA > 0) {
                aWorld = [
                    (pose.vNed.n - traj[i - 1].vNed.n) / dtA,
                    (pose.vNed.e - traj[i - 1].vNed.e) / dtA,
                    (pose.vNed.d - traj[i - 1].vNed.d) / dtA,
                ];
            }
        }
        // Specific force in world: sf_world = a_world − g_world
        const sfWorld = [aWorld[0], aWorld[1], aWorld[2] - g];
        // Rotate to body: sf_body = Rᵀ · sf_world
        const R = quatToRotMatrix(pose.q);
        const sfBody = [
            R[0][0] * sfWorld[0] + R[1][0] * sfWorld[1] + R[2][0] * sfWorld[2],
            R[0][1] * sfWorld[0] + R[1][1] * sfWorld[1] + R[2][1] * sfWorld[2],
            R[0][2] * sfWorld[0] + R[1][2] * sfWorld[1] + R[2][2] * sfWorld[2],
        ];
        // Raw accel = −sf_body (MEMS reads reaction force, +g at rest)
        const accel = [
            -sfBody[0] + (rng ? randn(rng, 0, accelNoiseStd) : 0),
            -sfBody[1] + (rng ? randn(rng, 0, accelNoiseStd) : 0),
            -sfBody[2] + (rng ? randn(rng, 0, accelNoiseStd) : 0),
        ];

        imu.push({ tUs, gyro, accel });

        // ---- GPS at ~10 Hz intervals ----
        if (i % (Math.round(1 / (10 * dt))) === 0) {            gpsList.push({
                tUs,
                lat: 48.408 + pose.pNed.n / 111320,
                lon: -71.164 + pose.pNed.e / (111320 * Math.cos(48.408 * Math.PI / 180)),
                alt: originAlt - pose.pNed.d,
                velNed: [
                    pose.vNed.n + (rng ? randn(rng, 0, 0.3) : 0),
                    pose.vNed.e + (rng ? randn(rng, 0, 0.3) : 0),
                    pose.vNed.d + (rng ? randn(rng, 0, 0.3) : 0),
                ],
                numSat: 12,
                fixType: 3,
            });
        }

        // ---- Baro ----
        baro.push({ tUs, alt: -pose.pNed.d });

        // ---- FC quaternion ----
        quat.push({ tUs, q: pose.q });

        // ---- Magnetometer (3-axis in body) ----
        const Rq = quatToRotMatrix(pose.q);
        const meB = [
            Rq[0][0] * mEarth[0] + Rq[1][0] * mEarth[1] + Rq[2][0] * mEarth[2],
            Rq[0][1] * mEarth[0] + Rq[1][1] * mEarth[1] + Rq[2][1] * mEarth[2],
            Rq[0][2] * mEarth[0] + Rq[1][2] * mEarth[1] + Rq[2][2] * mEarth[2],
        ];
        mag.push({
            tUs,
            meas: [meB[0] + mBody[0], meB[1] + mBody[1], meB[2] + mBody[2]],
        });
    }

    return { imu, gps: gpsList, baro, quat, mag };
}

// Internal quaternion helpers for sensor synthesis
function quatMult(a, b) {
    const [aw, ax, ay, az] = a;
    const [bw, bx, by, bz] = b;
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ];
}

function quatToRotVec(q) {
    const w = q[0], vNorm = Math.sqrt(q[1]**2 + q[2]**2 + q[3]**2);
    if (vNorm < 1e-14) return [0, 0, 0];
    const theta = 2 * Math.atan2(vNorm, w);
    const scale = theta / vNorm;
    return [q[1] * scale, q[2] * scale, q[3] * scale];
}
