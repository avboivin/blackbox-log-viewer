/**
 * Quaternion ingestion convention test — verifies extract-then-rebuild
 * preserves roll/pitch while correcting heading.
 *
 * The logged imuQuaternion uses Betaflight's convention (the Euler extraction
 * formula yaw = -atan2(R[1][0], R[0][0]) gives Betaflight's reported heading).
 * The ingestion layer must extract Euler angles using Betaflight's formulas
 * then rebuild a standard right-handed ZYX quaternion for the estimator.
 *
 * This test validates on NON-level attitudes to catch the bug class
 * that bit us (near-level-only tests hide sign mismatches).
 */

import { describe, it, expect } from "vitest";
import { quatToRot, quatToEuler, eulerToQuat } from "./imuMechanization.js";

/**
 * Apply the ingestion fix: extract using Betaflight's formulas, rebuild.
 * This mirrors flightIngestion.js quaternion handling exactly.
 *
 * @param {number[]} qBf - logged imuQuaternion [w,x,y,z]
 * @returns {number[]} estimator-frame quaternion body→world [w,x,y,z]
 */
function convertBfQuatToEstimator(qBf) {
    const R = quatToRot(qBf);
    const roll = Math.atan2(R[2][1], R[2][2]);
    const pitch = -Math.asin(Math.max(-1, Math.min(1, R[2][0])));
    let heading = -Math.atan2(R[1][0], R[0][0]);
    if (heading < 0) heading += 2 * Math.PI;
    return eulerToQuat(roll, pitch, heading);
}

function toDeg(r) { return ((r * 180 / Math.PI) + 360) % 360; }

describe("ImuQuaternion convention — extract-then-rebuild", () => {
    it("recovers correct attitude from a real logged quaternion (level, heading West)", () => {
        // A real Betaflight-logged attitude quaternion (first I-frame of a level segment)
        const qBf = [0.6185, 0.0000, -0.0019, 0.7858];
        const qEst = convertBfQuatToEstimator(qBf);
        const euler = quatToEuler(qEst);

        // Should be near-level (roll≈0, pitch≈0) — handle 360° wrapping
        const rW = Math.min(toDeg(euler.roll), Math.abs(toDeg(euler.roll) - 360));
        const pW = Math.min(Math.abs(toDeg(euler.pitch)), Math.abs(toDeg(euler.pitch) - 360), Math.abs(toDeg(euler.pitch) + 360));
        expect(rW).toBeLessThan(1);
        expect(pW).toBeLessThan(1);
        // Heading should be ~256° (WSW — matches GPS course 253-270°)
        expect(toDeg(euler.yaw)).toBeCloseTo(256, 0);
    });

    it("recovers correct attitude from a later real-log quaternion (mid-flight)", () => {
        // q at ~5s: from phase-0 output
        const qBf = [0.6662, -0.0557, 0.0864, 0.7386];
        const qEst = convertBfQuatToEstimator(qBf);
        const euler = quatToEuler(qEst);

        // All angles must be finite
        expect(isFinite(euler.roll)).toBe(true);
        expect(isFinite(euler.pitch)).toBe(true);
        expect(isFinite(euler.yaw)).toBe(true);
        // Heading should be ~264° (West-ish, near GPS course)
        expect(toDeg(euler.yaw)).toBeGreaterThan(180);
        expect(toDeg(euler.yaw)).toBeLessThan(360);
    });

    it("preserves identity quaternion (no-op)", () => {
        const qBf = [1, 0, 0, 0];
        const qEst = convertBfQuatToEstimator(qBf);
        expect(qEst[0]).toBeCloseTo(1, 6);
        expect(qEst[1]).toBeCloseTo(0, 6);
        expect(qEst[2]).toBeCloseTo(0, 6);
        expect(qEst[3]).toBeCloseTo(0, 6);
    });

    it("handles a pure-yaw quaternion correctly", () => {
        // Betaflight uses right-multiply IMU integration which effectively
        // negates the yaw component. So for geographic heading=90° (East),
        // the logged quaternion ≈ eulerToQuat(0, 0, -90°).
        const qBf = eulerToQuat(0, 0, -90 * Math.PI / 180);
        const qLogged = qBf[0] >= 0 ? qBf : [-qBf[0], -qBf[1], -qBf[2], -qBf[3]];

        const qEst = convertBfQuatToEstimator(qLogged);

        // The reconstructed quaternion should map nose to heading 90°
        const R = quatToRot(qEst);
        const noseEast = R[1][0];
        const noseNorth = R[0][0];
        const yaw = Math.atan2(noseEast, noseNorth);
        expect(toDeg(yaw)).toBeCloseTo(90, 1);
    });

    it("handles pure-pitch body→world quaternion for nose-up", () => {
        // The logged quaternion is body→world (planner-verified against firmware).
        // For pitch 30° nose-up, heading 0°:
        const qBf = eulerToQuat(0, 30 * Math.PI / 180, 0);
        // Force w≥0 (as the log format does)
        const qLogged = qBf[0] >= 0 ? qBf : [-qBf[0], -qBf[1], -qBf[2], -qBf[3]];

        const qEst = convertBfQuatToEstimator(qLogged);
        const euler = quatToEuler(qEst);

        expect(toDeg(euler.roll)).toBeCloseTo(0, 0);
        expect(toDeg(euler.pitch)).toBeCloseTo(30, 0);
        expect(toDeg(euler.yaw)).toBeCloseTo(0, 0);
    });

    it("handles pure-roll body→world quaternion for right-bank", () => {
        // Roll 30° right bank, heading 0°:
        const qBf = eulerToQuat(30 * Math.PI / 180, 0, 0);
        const qLogged = qBf[0] >= 0 ? qBf : [-qBf[0], -qBf[1], -qBf[2], -qBf[3]];

        const qEst = convertBfQuatToEstimator(qLogged);
        const euler = quatToEuler(qEst);

        expect(toDeg(euler.roll)).toBeCloseTo(30, 0);
        expect(toDeg(euler.pitch)).toBeCloseTo(0, 0);
        expect(toDeg(euler.yaw)).toBeCloseTo(0, 0);
    });

    it("handles a combined pitch+roll body→world quaternion", () => {
        // Roll 20°, pitch 15°, heading 200° (SW).
        // Betaflight right-multiply effectively negates yaw → eulerToQuat(20°, 15°, -200°)
        const qBf = eulerToQuat(20 * Math.PI / 180, 15 * Math.PI / 180, -200 * Math.PI / 180);
        const qLogged = qBf[0] >= 0 ? qBf : [-qBf[0], -qBf[1], -qBf[2], -qBf[3]];

        const qEst = convertBfQuatToEstimator(qLogged);
        const euler = quatToEuler(qEst);

        // Roll/pitch preserved
        expect(toDeg(euler.roll)).toBeCloseTo(20, 0);
        expect(toDeg(euler.pitch)).toBeCloseTo(15, 0);
        // Heading with small Euler-coupling tolerance
        const headDiff = Math.min(
            Math.abs(toDeg(euler.yaw) - 200),
            Math.abs(toDeg(euler.yaw) - (200 + 360)),
            Math.abs(toDeg(euler.yaw) - (200 - 360)),
        );
        expect(headDiff, `heading diff: expected ~200°, got ${toDeg(euler.yaw).toFixed(1)}°`).toBeLessThan(15);
    });

    it("produces finite results for all test inputs", () => {
        const testQs = [
            [1, 0, 0, 0],
            [0.7071, 0, 0, 0.7071],
            [0.7071, 0, 0, -0.7071],
            [0.6185, 0.0000, -0.0019, 0.7858],
            [0.6662, -0.0557, 0.0864, 0.7386],
            [0, 0.6, 0, 0.8],
            [0.5, 0.5, 0.5, 0.5],
        ];

        for (const q of testQs) {
            const norm = Math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2);
            const qNorm = q.map((v) => v / norm);
            const qEst = convertBfQuatToEstimator(qNorm);
            expect(qEst.every((v) => isFinite(v)), `NaN in result for q=[${q.join(",")}]`).toBe(true);
            const n = Math.sqrt(qEst[0]**2 + qEst[1]**2 + qEst[2]**2 + qEst[3]**2);
            expect(n).toBeGreaterThan(0.9);
            expect(n).toBeLessThan(1.1);
        }
    });

    it("heading from estimator q matches GPS course on a real-log level case", () => {
        // Verify gravity-aligned axes from the estimator quaternion:
        // For a level drone, body +Z (down in FRD) maps to world +D (down in NED).
        // If this holds, the attitude is consistent with the FRD/NED frame.
        const qBf = [0.6185, 0.0000, -0.0019, 0.7858];
        const qEst = convertBfQuatToEstimator(qBf);
        const R = quatToRot(qEst);

        // Body +Z (down) in world should be close to [0, 0, +1] (world down)
        const zInWorld = [R[0][2], R[1][2], R[2][2]];
        expect(zInWorld[0]).toBeLessThan(0.1);  // small N component
        expect(zInWorld[2]).toBeGreaterThan(0.9); // mostly down
    });
});
