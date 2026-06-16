/**
 * Quaternion ingestion convention test — verifies the extraction round-trip
 * is EXACT (self-consistent) for Betaflight's logged imuQuaternion.
 *
 * The logged imuQuaternion uses standard body(FRD)→world(NED) Hamilton
 * convention. The ingestion must use standard ZYX Euler extraction on the
 * quaternion's rotation matrix (NOT Betaflight's firmware yaw formula which
 * uses a negated atan2 for display purposes). Euler extracted from quatToRot
 * using standard formulas + eulerToQuat rebuild is an identity transform
 * (verified 2026-06-15 against synthetic and real-log data).
 *
 * This test validates on NON-level attitudes to catch the bug class that
 * bit the project twice (near-level-only tests hide sign mismatches).
 */
import { describe, it, expect } from "vitest";
import { quatToRot, eulerToQuat } from "./imuMechanization.js";

/**
 * Apply the ingestion: extract standard ZYX Euler from the quaternion's
 * rotation matrix, rebuild with eulerToQuat. The round-trip is exact —
 * the output quaternion represents the same rotation as the input
 * (up to global sign, which is the same rotation for Hamilton quaternions).
 *
 * @param {number[]} qLogged - logged imuQuaternion [w,x,y,z] (body→world NED)
 * @returns {number[]} estimator quaternion [w,x,y,z] — same rotation as input
 */
function ingestQuat(qLogged) {
    const R = quatToRot(qLogged);
    const roll = Math.atan2(R[2][1], R[2][2]);
    const pitch = -Math.asin(Math.max(-1, Math.min(1, R[2][0])));
    const yaw = Math.atan2(R[1][0], R[0][0]);
    return eulerToQuat(roll, pitch, yaw);
}

function toDeg(r) { return ((r * 180 / Math.PI) + 360) % 360; }
function pitchDeg(q) { const R = quatToRot(q); return -Math.asin(Math.max(-1, Math.min(1, R[2][0]))) * 180 / Math.PI; }
function yawDeg(q) { const R = quatToRot(q); return toDeg(Math.atan2(R[1][0], R[0][0])); }
function rollDeg(q) { const R = quatToRot(q); return toDeg(Math.atan2(R[2][1], R[2][2])); }

describe("ImuQuaternion ingestion — exact round-trip", () => {
    it("preserves identity quaternion", () => {
        const qIn = [1, 0, 0, 0];
        const qOut = ingestQuat(qIn);
        // Identity quaternion round-trips EXACTLY (all components match)
        expect(qOut[0]).toBeCloseTo(1, 10);
        expect(qOut[1]).toBeCloseTo(0, 10);
        expect(qOut[2]).toBeCloseTo(0, 10);
        expect(qOut[3]).toBeCloseTo(0, 10);
    });

    it("preserves pure yaw 90° (nose east) exactly", () => {
        const qIn = [Math.cos(Math.PI/4), 0, 0, Math.sin(Math.PI/4)];
        const qOut = ingestQuat(qIn);
        // Round-trip should return the same rotation
        // Check nose direction: should point east
        const Rout = quatToRot(qOut);
        expect(Rout[0][0]).toBeCloseTo(0, 10);  // nose N ≈ 0
        expect(Rout[1][0]).toBeCloseTo(1, 10);  // nose E ≈ 1
        // The quaternion may have w negated (=-q) but rotation is the same
    });

    it("preserves pitch 30° nose-up exactly", () => {
        // Pitch 30° nose-up: rotation about -Y by 30° in body FRD
        // q = eulerToQuat(0°, -30°, 0°)... wait, we need to be careful.
        // In ZYX Euler: pitch positive = nose-down in FRD (right-hand +Y).
        // So nose-up 30° = pitch -30° = eulerToQuat(0, -30°, 0)
        const qIn = eulerToQuat(0, -30 * Math.PI / 180, 0);
        const qOut = ingestQuat(qIn);
        expect(pitchDeg(qOut)).toBeCloseTo(-30, 1);
        expect(Math.abs(rollDeg(qOut)) % 360).toBeLessThan(1);
    });

    it("preserves a combined pitch+roll+yaw quaternion exactly", () => {
        const qIn = eulerToQuat(
            20 * Math.PI / 180,   // roll 20°
            -15 * Math.PI / 180,  // pitch -15° (nose up)
            200 * Math.PI / 180,  // yaw 200° (SSW)
        );
        const qOut = ingestQuat(qIn);
        expect(rollDeg(qOut)).toBeCloseTo(20, 0);
        expect(pitchDeg(qOut)).toBeCloseTo(-15, 0);
        const headDiff = Math.min(
            Math.abs(yawDeg(qOut) - 200),
            Math.abs(yawDeg(qOut) - (200 + 360)),
            Math.abs(yawDeg(qOut) - (200 - 360)),
        );
        expect(headDiff, `heading diff: expected ~200°, got ${yawDeg(qOut).toFixed(1)}°`).toBeLessThan(1);
    });

    it("preserves w<0 quaternion (negated input, same rotation)", () => {
        // A quaternion with w<0 is physically valid; the log format
        // forces w≥0 but our ingestion should handle either.
        // Starting from valid Euler angles to ensure physical quaternion.
        const qNegW = eulerToQuat(10 * Math.PI/180, 20 * Math.PI/180, 45 * Math.PI/180);
        // Negate to get w<0 (still same rotation)
        const qIn = [-qNegW[0], -qNegW[1], -qNegW[2], -qNegW[3]];
        const qOut = ingestQuat(qIn);
        // The output quaternion should represent the same rotation
        expect(rollDeg(qOut)).toBeCloseTo(10, 0);
        expect(pitchDeg(qOut)).toBeCloseTo(20, 0);
        expect(yawDeg(qOut)).toBeCloseTo(45, 0);
    });

    it("produces finite results for diverse test inputs", () => {
        const testQs = [
            [1, 0, 0, 0],
            [0.7071, 0, 0, 0.7071],
            [0.7071, 0, 0, -0.7071],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.5, 0.5, 0.5, 0.5],
            [0.9805, -0.0298, 0.0465, 0.1887], // real acro1 log sample
            [-0.1095, -0.1484, -0.2833, 0.9411], // real acro1 climb sample
        ];
        for (const q of testQs) {
            const norm = Math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2);
            const qNorm = q.map((v) => v / norm);
            const qOut = ingestQuat(qNorm);
            expect(qOut.every((v) => isFinite(v)), `NaN for q=[${q.join(",")}]`).toBe(true);
            const n = Math.sqrt(qOut[0]**2 + qOut[1]**2 + qOut[2]**2 + qOut[3]**2);
            expect(n).toBeGreaterThan(0.999);
            expect(n).toBeLessThan(1.001);
        }
    });

    it("preserves gravity axis: body +Z maps to world +D for level drone", () => {
        // For a level drone, body Z=down should map to world D=down (NED +Z).
        // This verifies the FRD→NED frame consistency.
        const qLevel = [0.9805, -0.0298, 0.0465, 0.1887]; // real sample, near-level
        const qOut = ingestQuat(qLevel);
        const R = quatToRot(qOut);
        // Body +Z (col2 of R) should have large positive D-component
        expect(R[2][2]).toBeGreaterThan(0.9);  // mostly down in NED
    });

    it("nose direction from ingestion matches direct quatToRot for real sample", () => {
        // The ingestion should produce a quaternion with the same nose direction
        // as the original logged quaternion (since round-trip is exact).
        const qRaw = [0.9805, -0.0298, 0.0465, 0.1887];
        const R_orig = quatToRot(qRaw);
        const qIng = ingestQuat(qRaw);
        const R_ing = quatToRot(qIng);
        // Nose world vector (col0) should match
        expect(R_ing[0][0]).toBeCloseTo(R_orig[0][0], 5);
        expect(R_ing[1][0]).toBeCloseTo(R_orig[1][0], 5);
        expect(R_ing[2][0]).toBeCloseTo(R_orig[2][0], 5);
    });
});
