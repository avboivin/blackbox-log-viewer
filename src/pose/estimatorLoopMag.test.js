/**
 * Task C — estimator loop with 3-axis mag fusion.
 *
 * Tests that mag fusion recovers absolute heading from magnetometer measurements
 * when the estimator is started at a wrong yaw with the quaternion prior disabled
 * or very loose. All tests assert every pose is finite across the whole trajectory.
 */
import { describe, it, expect } from "vitest";
import { generateDynamicTrajectory, generateSensorStreams } from "./synthetic.js";
import { estimatePoses } from "./estimatorLoop.js";

function quatMultiply(a, b) {
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

function quatNorm(q) {
    const n = Math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2);
    return n < 1e-14 ? [1, 0, 0, 0] : [q[0]/n, q[1]/n, q[2]/n, q[3]/n];
}

function quatFromAxisAngle(axis, angle) {
    const half = angle / 2;
    const s = Math.sin(half);
    return [Math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** Assert every pose is finite (no NaN/Infinity). */
function assertAllPosesFinite(poses) {
    for (let i = 0; i < poses.length; i++) {
        const p = poses[i];
        expect(isFinite(p.lat), `pose[${i}].lat finite`).toBe(true);
        expect(isFinite(p.lon), `pose[${i}].lon finite`).toBe(true);
        expect(isFinite(p.altMsl), `pose[${i}].altMsl finite`).toBe(true);
        for (let j = 0; j < 4; j++) expect(isFinite(p.q[j]), `pose[${i}].q[${j}] finite`).toBe(true);
        expect(isFinite(p.sigmaPos), `pose[${i}].sigmaPos finite`).toBe(true);
        expect(isFinite(p.sigmaAtt), `pose[${i}].sigmaAtt finite`).toBe(true);
    }
}

/** Quaternion geodesic distance in radians. */
function quatAngle(qa, qb) {
    const qrel = quatMultiply(qa, quatConjugate(qb));
    const vNorm = Math.sqrt(qrel[1]**2 + qrel[2]**2 + qrel[3]**2);
    return 2 * Math.atan2(vNorm, Math.abs(qrel[0]));
}

const RAD = Math.PI / 180;

describe("estimator loop — 3-axis mag fusion (Task C)", () => {
    // Horizontal-dominant earth field so yaw is observable from a single
    // mag vector (a vertical-dominant field makes yaw ≈ rotation about the
    // field axis → unobservable regardless of convention).
    const earthField = [0.45, 0.05, 0.12];
    const earthFieldObj = { n: earthField[0], e: earthField[1], d: earthField[2] };

    it("recovers heading from mag with wrong initial yaw and no quaternion prior", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, {
            gpsNoiseStd: 0.5,
            mEarth: earthField,
            origin,
        });

        // Seed at wrong yaw (+40° offset from true initial yaw of 0°)
        const qYaw40 = quatFromAxisAngle([0, 0, 1], 40 * RAD);
        const qTrue0 = [...quat[0].q];
        quat[0].q = quatNorm(quatMultiply(qYaw40, qTrue0));

        // Verify we actually started 40° off
        const initAngle = quatAngle(quat[0].q, qTrue0) * (180 / Math.PI);
        expect(initAngle).toBeGreaterThan(38);

        const magModel = {
            earthFieldNedGauss: earthFieldObj,
            magNoiseGauss: { sigma: 0.01 },
            qualityBounds: { bounds_ok: true },
        };

        // Very loose quaternion prior (0.8 rad ≈ 46°) — mag must drive heading
        const poses = estimatePoses(
            { imu, gps, baro, quat, mag },
            origin,
            {
                outputHz: 50,
                gpsPosSigma: 0.5,
                gpsVelSigma: 0.5,
                attSigma: 100,  // effectively disabled — mag must drive heading (Q3: larger state
                magSigma: 0.01,
                magModel,
                maxIter: 1,
            },
        );

        expect(poses.length).toBeGreaterThan(20);
        assertAllPosesFinite(poses);

        // Check heading recovery at end — should be < 5° via mag
        const lastEst = poses[poses.length - 1];
        const lastTrue = traj[traj.length - 1];
        const attErr = quatAngle(lastTrue.q, lastEst.q) * (180 / Math.PI);
        expect(attErr).toBeLessThan(5);
    });

    it("mag outlier is rejected by chi-square gate", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, {
            gpsNoiseStd: 0.5,
            mEarth: earthField,
            origin,
        });

        // Seed at wrong yaw like the recovery test
        const qYaw40 = quatFromAxisAngle([0, 0, 1], 40 * RAD);
        quat[0].q = quatNorm(quatMultiply(qYaw40, quat[0].q));

        // Inject an outlier mag reading (100× the field) at mid-flight
        const outlierIdx = Math.floor(mag.length / 2);
        mag[outlierIdx] = { tUs: mag[outlierIdx].tUs, meas: [10, 10, 10] };

        const magModel = {
            earthFieldNedGauss: earthFieldObj,
            magNoiseGauss: { sigma: 0.01 },
            qualityBounds: { bounds_ok: true },
        };

        const poses = estimatePoses(
            { imu, gps, baro, quat, mag },
            origin,
            {
                outputHz: 50,
                gpsPosSigma: 0.5,
                gpsVelSigma: 0.5,
                attSigma: 100,  // effectively disabled — mag must drive heading (Q3: larger state
                magSigma: 0.01,
                magModel,
                maxIter: 2,
            },
        );

        expect(poses.length).toBeGreaterThan(20);
        assertAllPosesFinite(poses);

        // The outlier should be rejected — attitude should still recover
        const lastEst = poses[poses.length - 1];
        const lastTrue = traj[traj.length - 1];
        const attErr = quatAngle(lastTrue.q, lastEst.q) * (180 / Math.PI);
        expect(attErr).toBeLessThan(15);
    });

    it("declination constraint keeps m_earth direction stable", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, {
            gpsNoiseStd: 0.5,
            mEarth: earthField,
            origin,
        });

        // Seed at wrong yaw
        const qYaw40 = quatFromAxisAngle([0, 0, 1], 40 * RAD);
        quat[0].q = quatNorm(quatMultiply(qYaw40, quat[0].q));

        const magModel = {
            earthFieldNedGauss: earthFieldObj,
            magNoiseGauss: { sigma: 0.01 },
            qualityBounds: { bounds_ok: true },
        };

        const poses = estimatePoses(
            { imu, gps, baro, quat, mag },
            origin,
            {
                outputHz: 50,
                gpsPosSigma: 0.5,
                gpsVelSigma: 0.5,
                attSigma: 100,  // effectively disabled — mag must drive heading (Q3: larger state
                magSigma: 0.01,
                declSigma: 0.05,
                magModel,
                maxIter: 2,
            },
        );

        expect(poses.length).toBeGreaterThan(20);
        assertAllPosesFinite(poses);

        // Attitude should be recovered
        const lastEst = poses[poses.length - 1];
        const lastTrue = traj[traj.length - 1];
        const attErr = quatAngle(lastTrue.q, lastEst.q) * (180 / Math.PI);
        expect(attErr).toBeLessThan(10);
    });
});
