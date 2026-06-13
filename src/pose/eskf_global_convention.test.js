import { describe, it, expect } from "vitest";
import { createEskf, eskfUpdate } from "./eskf.js";
import { createMagFactor, createQuaternionPrior } from "./measurements.js";
import { quatMultiply, quatFromAxisAngle, quatToEuler, quatToRot } from "./imuMechanization.js";

// ---------------------------------------------------------------------------
// Regression tests for the GLOBAL (world-frame) attitude-error convention.
//
// The estimator defines  q_true = δq(δθ_world) ⊗ q̂  (06 §1). The predict F,
// the mag Jacobian H_θ, the quaternion-prior residual, and the state injection
// must ALL use this convention. A prior implementation mixed a global injection
// with local/body Jacobians; that bug is invisible at identity attitude
// (where local ≡ global) but corrupts any tilted, rotating segment.
//
// These tests start from a NON-identity attitude with a wrong yaw and require
// the filter to rotate to truth. They fail on the mixed-convention code and
// pass only when every site is consistently global.
// ---------------------------------------------------------------------------

/** ZYX intrinsic Euler (yaw Z, pitch Y, roll X) → quaternion, body(FRD)→world(NED). */
function eulerToQuat(yaw, pitch, roll) {
    const qz = quatFromAxisAngle([0, 0, 1], yaw);
    const qy = quatFromAxisAngle([0, 1, 0], pitch);
    const qx = quatFromAxisAngle([1, 0, 0], roll);
    return quatMultiply(qz, quatMultiply(qy, qx));
}

const D2R = Math.PI / 180;
const R2D = 180 / Math.PI;

function yawErrDeg(qa, qb) {
    let d = (quatToEuler(qa).yaw - quatToEuler(qb).yaw) * R2D;
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    return Math.abs(d);
}

describe("ESKF global attitude-error convention (tilted-attitude regression)", () => {
    it("recovers yaw from 3-axis mag at a NON-level attitude", () => {
        // True attitude: meaningfully tilted AND yawed.
        const qTrue = eulerToQuat(40 * D2R, 15 * D2R, 10 * D2R);

        // Earth field in NED (Gauss): horizontal-dominant so a single mag vector
        // observes YAW (a vertical-dominant field makes yaw ≈ rotation about the
        // field axis → unobservable from one vector, regardless of convention).
        const mEarth = [0.45, 0.05, 0.12];

        // Perfect, iron-free body reading: z = R_true^T · m_earth
        const Rt = quatToRot(qTrue); // body→world
        const magMeas = [
            Rt[0][0] * mEarth[0] + Rt[1][0] * mEarth[1] + Rt[2][0] * mEarth[2],
            Rt[0][1] * mEarth[0] + Rt[1][1] * mEarth[1] + Rt[2][1] * mEarth[2],
            Rt[0][2] * mEarth[0] + Rt[1][2] * mEarth[1] + Rt[2][2] * mEarth[2],
        ];

        // Estimate starts with the SAME tilt but a wrong yaw (off by 40°).
        const qEst0 = eulerToQuat(0, 15 * D2R, 10 * D2R);

        const eskf = createEskf({
            p0: [0, 0, -100],
            v0: [0, 0, 0],
            q0: qEst0,
            mEarth0: mEarth,          // earth field seeded correctly…
            mBody0: [0, 0, 0],
            sigmaMagEarth: 0.0005,    // …and held tight, so residual loads onto attitude
            sigmaMagBody: 0.0005,
            sigmaAtt: 0.6,            // attitude free to move
        });

        const before = yawErrDeg(eskf.q, qTrue);
        expect(before).toBeGreaterThan(35); // we really did start ~40° off

        // Repeated mag updates (pure correction, no predict).
        for (let i = 0; i < 60; i++) {
            const factor = createMagFactor(magMeas, 0.002);
            eskfUpdate(eskf, factor, magMeas, 1e6); // wide gate; we test convergence, not gating
        }

        const after = yawErrDeg(eskf.q, qTrue);
        expect(after).toBeLessThan(3.0); // converged to true yaw
    });

    it("recovers attitude from the quaternion prior at a NON-level attitude", () => {
        const qTrue = eulerToQuat(-35 * D2R, 20 * D2R, -12 * D2R);
        const qEst0 = eulerToQuat(10 * D2R, 20 * D2R, -12 * D2R); // 45° yaw off

        const eskf = createEskf({
            p0: [0, 0, -100],
            v0: [0, 0, 0],
            q0: qEst0,
            sigmaAtt: 0.8,
        });

        const before = yawErrDeg(eskf.q, qTrue);
        expect(before).toBeGreaterThan(40);

        for (let i = 0; i < 40; i++) {
            const factor = createQuaternionPrior(qTrue, 0.05);
            eskfUpdate(eskf, factor, qTrue, 1e6);
        }

        const after = yawErrDeg(eskf.q, qTrue);
        expect(after).toBeLessThan(2.0);

        // full attitude (not just yaw): rotation angle between est and true is small
        const qrel = quatMultiply(qTrue, [eskf.q[0], -eskf.q[1], -eskf.q[2], -eskf.q[3]]);
        const angle = 2 * Math.acos(Math.min(1, Math.abs(qrel[0]))) * R2D;
        expect(angle).toBeLessThan(2.0);
    });
});
