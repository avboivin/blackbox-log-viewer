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
