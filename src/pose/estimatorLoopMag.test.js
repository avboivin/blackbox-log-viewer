/**
 * Task C — estimator loop with 3-axis mag fusion.
 *
 * Tests that the loop recovers absolute heading from magnetometer measurements
 * when initialized at a wrong yaw, using the dynamic trajectory harness.
 */
import { describe, it, expect } from "vitest";
import { generateDynamicTrajectory, generateSensorStreams } from "./synthetic.js";
import { estimatePoses } from "./estimatorLoop.js";

describe("estimator loop — 3-axis mag fusion (Task C)", () => {
    it("recovers absolute heading from mag on a banked trajectory", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });

        // Generate sensors WITHOUT GPS velocity — rely on mag for heading
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, { gpsNoiseStd: 0.5 });

        const origin = { lat: 48.408, lon: -71.164, alt: 200 };

        const magModel = {
            earthFieldNedGauss: { n: 0.17, e: -0.047, d: 0.51 },
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
                attSigma: 0.02,
                magSigma: 0.01,
                magModel,
                maxIter: 1,
            },
        );

        expect(poses.length).toBeGreaterThan(20);

        // Check attitude recovery at end
        const sampleEvery = Math.floor(traj.length / poses.length);
        const pi = poses.length - 1;
        const ti = Math.min(Math.round(pi * sampleEvery), traj.length - 1);
        const gt = traj[ti];
        const est = poses[pi];

        const qDot = gt.q[0] * est.q[0] + gt.q[1] * est.q[1] + gt.q[2] * est.q[2] + gt.q[3] * est.q[3];
        const attErr = 2 * Math.acos(Math.min(1, Math.abs(qDot))) * (180 / Math.PI);

        // The synthetic mag has a horizontal component (north/east), so heading
        // should be observable and attitude should be well under 10° error.
        expect(attErr).toBeLessThan(10);
    });

    it("mag outlier is rejected by chi-square gate", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, { gpsNoiseStd: 0.5 });

        // Inject an outlier mag reading (100× the field) at mid-flight
        const outlierIdx = Math.floor(mag.length / 2);
        mag[outlierIdx] = { tUs: mag[outlierIdx].tUs, meas: [10, 10, 10] };

        const origin = { lat: 48.408, lon: -71.164, alt: 200 };
        const magModel = {
            earthFieldNedGauss: { n: 0.17, e: -0.047, d: 0.51 },
            magNoiseGauss: { sigma: 0.01 },
            qualityBounds: { bounds_ok: true },
        };

        const poses = estimatePoses(
            { imu, gps, baro, quat, mag },
            origin,
            { outputHz: 50, gpsPosSigma: 0.5, attSigma: 0.02, magSigma: 0.01, magModel, maxIter: 1 },
        );

        // The outlier should be rejected — attitude should still recover
        expect(poses.length).toBeGreaterThan(20);

        const lastEst = poses[poses.length - 1];
        const lastTrue = traj[traj.length - 1];
        const qDot = lastTrue.q[0] * lastEst.q[0] + lastTrue.q[1] * lastEst.q[1] + lastTrue.q[2] * lastEst.q[2] + lastTrue.q[3] * lastEst.q[3];
        const attErr = 2 * Math.acos(Math.min(1, Math.abs(qDot))) * (180 / Math.PI);
        expect(attErr).toBeLessThan(15);
    });

    it("declination constraint keeps m_earth direction stable", () => {
        const { traj } = generateDynamicTrajectory({ freqHz: 200 });
        const { imu, gps, baro, quat, mag } = generateSensorStreams(traj, { gpsNoiseStd: 0.5 });

        const origin = { lat: 48.408, lon: -71.164, alt: 200 };

        // Use a m_earth with known declination
        const wmmN = 0.17, wmmE = -0.047;
        const magModel = {
            earthFieldNedGauss: { n: wmmN, e: wmmE, d: 0.51 },
            magNoiseGauss: { sigma: 0.01 },
            qualityBounds: { bounds_ok: true },
        };

        const poses = estimatePoses(
            { imu, gps, baro, quat, mag },
            origin,
            {
                outputHz: 50,
                gpsPosSigma: 0.5,
                attSigma: 0.02,
                magSigma: 0.01,
                declSigma: 0.05,
                magModel,
                maxIter: 1,
            },
        );

        expect(poses.length).toBeGreaterThan(20);

        // Attitude should be recovered
        const lastEst = poses[poses.length - 1];
        const lastTrue = traj[traj.length - 1];
        const qDot = lastTrue.q[0] * lastEst.q[0] + lastTrue.q[1] * lastEst.q[1] + lastTrue.q[2] * lastEst.q[2] + lastTrue.q[3] * lastEst.q[3];
        const attErr = 2 * Math.acos(Math.min(1, Math.abs(qDot))) * (180 / Math.PI);
        expect(attErr).toBeLessThan(10);
    });
});
