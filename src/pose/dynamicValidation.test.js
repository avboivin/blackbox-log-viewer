/**
 * Non-level validation harness — dynamic trajectory round-trip test.
 *
 * Verifies the estimator recovers ground-truth pose from synthesized sensors
 * on a trajectory with sustained non-zero roll, pitch, and yaw rate.
 * Identity-attitude tests cannot catch frame/rotation convention bugs.
 */
import { describe, it, expect } from "vitest";
import { generateDynamicTrajectory, generateSensorStreams } from "./synthetic.js";
import { estimatePoses } from "./estimatorLoop.js";

describe("dynamic trajectory round-trip (Task A)", () => {
    it("recovers position and attitude on a banked-turn-climb-spin trajectory", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const { imu, gps, baro, quat } = generateSensorStreams(traj, { gpsNoiseStd: 0.5, origin });

        const poses = estimatePoses(
            { imu, gps, baro, quat },
            origin,
            { outputHz: 50, gpsPosSigma: 0.5, gpsVelSigma: 0.3, baroSigma: 0.5, attSigma: 0.05, maxIter: 1 },
        );

        expect(poses.length).toBeGreaterThan(20);

        // Compute mean position error over the trajectory
        const sampleEvery = Math.floor(traj.length / poses.length);
        let totalPosErr = 0, totalAttErr = 0, maxPosErr = 0, count = 0;

        for (let pi = 0; pi < poses.length; pi++) {
            const ti = Math.min(Math.round(pi * sampleEvery), traj.length - 1);
            const gt = traj[ti];
            const est = poses[pi];

            const posErr = Math.sqrt(
                ((est.lat - 48.408) * 111320 - gt.pNed.n) ** 2 +
                ((est.lon + 71.164) * 111320 * Math.cos(48.408 * Math.PI / 180) - gt.pNed.e) ** 2 +
                (-(est.altMsl - origin.alt) - gt.pNed.d) ** 2,
            );

            totalPosErr += posErr;
            maxPosErr = Math.max(maxPosErr, posErr);

            // Attitude error via quaternion difference
            const qErr = [
                gt.q[0] * est.q[0] + gt.q[1] * est.q[1] + gt.q[2] * est.q[2] + gt.q[3] * est.q[3],
                -gt.q[1] * est.q[0] + gt.q[0] * est.q[1] - gt.q[3] * est.q[2] + gt.q[2] * est.q[3],
                -gt.q[2] * est.q[0] + gt.q[3] * est.q[1] + gt.q[0] * est.q[2] - gt.q[1] * est.q[3],
                -gt.q[3] * est.q[0] - gt.q[2] * est.q[1] + gt.q[1] * est.q[2] + gt.q[0] * est.q[3],
            ];
            const vNorm = Math.sqrt(qErr[1]**2 + qErr[2]**2 + qErr[3]**2);
            const attDeg = 2 * Math.atan2(vNorm, Math.abs(qErr[0])) * (180 / Math.PI);
            totalAttErr += attDeg;
            count++;
        }

        const meanPosErr = totalPosErr / count;
        const meanAttErr = totalAttErr / count;

        // With GPS at 0.5m sigma + quaternion prior at 0.05 rad, position should be < 3m
        expect(meanPosErr).toBeLessThan(3.0);
        // Attitude should be < 5° on average
        expect(meanAttErr).toBeLessThan(5.0);
        // No single point should be catastrophic (>20m)
        expect(maxPosErr).toBeLessThan(20.0);
    });

    it("generates self-consistent sensor streams (level static check)", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const { imu, gps, baro, mag } = generateSensorStreams(traj, { origin });

        // First sample: the trajectory starts with a banked attitude right away,
        // so the accel Z should not be exactly 9.81.
        // But the stream should have data for every IMU sample.
        expect(imu.length).toBe(traj.length);
        expect(imu[0].gyro.length).toBe(3);
        expect(imu[0].accel.length).toBe(3);

        // GPS should appear at intervals
        expect(gps.length).toBeGreaterThan(5);

        // Baro should be present
        expect(baro.length).toBe(traj.length);

        // Mag should be present for every step
        expect(mag.length).toBe(traj.length);
        // For the banked attitude, mag body vector should NOT equal earth vector
        const mEarth = [0.17, -0.047, 0.51];
        const firstMag = mag[0].meas;
        const distFromEarth = Math.sqrt(
            (firstMag[0] - mEarth[0])**2 + (firstMag[1] - mEarth[1])**2 + (firstMag[2] - mEarth[2])**2,
        );
        // Should differ because body is banked → Rᵀ transforms m_earth
        expect(distFromEarth).toBeGreaterThan(0.01);
    });
});
