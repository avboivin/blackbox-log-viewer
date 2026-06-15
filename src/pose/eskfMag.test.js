import { describe, it, expect } from "vitest";
import { createEskf, eskfUpdate } from "./eskf.js";
import { createMagFactor, createDeclinationFactor } from "./measurements.js";

describe("eskf — 21-state with mag fusion (Q3: unconditional 15-state base + 6 mag)", () => {
    it("initializes with magnetic field states", () => {
        const eskf = createEskf({
            p0: [0, 0, -200],
            v0: [0, 0, 0],
            q0: [1, 0, 0, 0],
            mEarth0: [0.17, -0.047, 0.51],
            mBody0: [0, 0, 0],
        });

        expect(eskf.dim).toBe(21);      // 15 base (inc. unconditional b_a/b_g) + 6 mag
        expect(eskf.mEarth).toEqual([0.17, -0.047, 0.51]);
        expect(eskf.mBody).toEqual([0, 0, 0]);
        expect(eskf.P.length).toBe(21);
        expect(eskf.P[0].length).toBe(21);
        // Bias states are always present (Q3)
        expect(eskf.ba).toEqual([0, 0, 0]);
        expect(eskf.bg).toEqual([0, 0, 0]);
    });

    it("15-state base ESKF works (no mag; unconditional b_a/b_g, Q3)", () => {
        const eskf = createEskf({
            p0: [0, 0, -200],
            v0: [7, 7, 0],
            q0: [1, 0, 0, 0],
        });

        expect(eskf.dim).toBe(15);      // unconditional b_a/b_g always present
        expect(eskf.mEarth).toBeNull();
        expect(eskf.P.length).toBe(15);
        expect(eskf.ba).toEqual([0, 0, 0]);
        expect(eskf.bg).toEqual([0, 0, 0]);
    });

    it("mag factor updates m_earth toward measurement", () => {
        const eskf = createEskf({
            p0: [0, 0, -200],
            v0: [0, 0, 0],
            q0: [1, 0, 0, 0],
            mEarth0: [0.17, -0.047, 0.51],
            mBody0: [0, 0, 0],
            sigmaMagEarth: 0.05,
        });

        // Simulate level drone, mag reads earth field in body = [0.17, -0.047, 0.51]
        // (in FRD body, which equals NED for level drone)
        const magMeas = [0.18, -0.05, 0.50];
        const factor = createMagFactor(magMeas, 0.01);
        const accepted = eskfUpdate(eskf, factor, magMeas);

        expect(accepted).toBe(true);

        // m_earth should have moved slightly toward the measurement
        expect(eskf.mEarth[0]).toBeGreaterThan(0.17);
        expect(eskf.mEarth[0]).toBeLessThan(0.18);
    });

    it("declination factor constrains m_earth direction", () => {
        const eskf = createEskf({
            p0: [0, 0, -200],
            v0: [0, 0, 0],
            q0: [1, 0, 0, 0],
            mEarth0: [0.18, -0.04, 0.51],
            mBody0: [0, 0, 0],
            sigmaMagEarth: 0.05,
        });

        // WMM declination at this location: atan2(-0.047, 0.17) ≈ -0.27 rad
        // Current declination: atan2(-0.04, 0.18) ≈ -0.218 rad
        // Measurement: -0.27 rad
        const declMeas = -0.27;
        const factor = createDeclinationFactor(declMeas, 0.1);
        const accepted = eskfUpdate(eskf, factor, declMeas);

        expect(accepted).toBe(true);
        // m_earth[1] (east) should become more negative (closer to -0.047)
        expect(eskf.mEarth[1]).toBeLessThan(-0.04);
    });

    it("mag outlier is rejected by chi-square gate", () => {
        const eskf = createEskf({
            p0: [0, 0, -200],
            v0: [0, 0, 0],
            q0: [1, 0, 0, 0],
            mEarth0: [0.17, -0.047, 0.51],
            mBody0: [0, 0, 0],
            sigmaMagEarth: 0.01,
            sigmaMagBody: 0.01,
        });

        // Converge first with a few good measurements
        const factor = createMagFactor([0.17, -0.047, 0.51], 0.001);
        for (let i = 0; i < 5; i++) {
            eskfUpdate(eskf, factor, [0.17, -0.047, 0.51]);
        }

        // Now inject a massive outlier (10 Gauss — 20× the field)
        const outlier = [10, 0, 0];
        const outlierFactor = createMagFactor(outlier, 0.01);
        const accepted = eskfUpdate(eskf, outlierFactor, outlier);

        expect(accepted).toBe(false);
        // m_earth should not have changed much
        expect(Math.abs(eskf.mEarth[0] - 0.17)).toBeLessThan(0.02);
    });
});
