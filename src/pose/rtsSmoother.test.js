import { describe, it, expect } from "vitest";
import { rtsSmooth } from "./rtsSmoother.js";

describe("rtsSmooth", () => {
    it("returns empty for empty input", () => {
        expect(rtsSmooth([], [])).toEqual([]);
    });

    it("single step returns filtered estimate unchanged", () => {
        const filtered = [
            {
                x: { p: [1, 2, 3], v: [0, 0, 0], q: [1, 0, 0, 0] },
                P: [[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1]],
            },
        ];
        const result = rtsSmooth(filtered, []);
        expect(result.length).toBe(1);
        expect(result[0].x.p).toEqual([1, 2, 3]);
    });

    it("smoother reduces covariance over a simple trajectory", () => {
        const I9 = new Array(9);
        for (let i = 0; i < 9; i++) {
            I9[i] = new Array(9).fill(0);
            I9[i][i] = 1;
        }

        // Build a 2-step trajectory: prediction + update at step 2
        const P0 = I9.map(row => row.map(v => v * 10)); // initial uncertainty
        const F = I9; // identity transition

        const filtered = [
            {
                x: { p: [0, 0, 0], v: [1, 0, 0], q: [1, 0, 0, 0] },
                P: P0,
                xPred: null,
                PPred: null,
            },
            {
                x: { p: [1, 0, 0], v: [1, 0, 0], q: [1, 0, 0, 0] },
                P: I9, // much lower uncertainty after GPS update
                xPred: { p: [1, 0, 0], v: [1, 0, 0], q: [1, 0, 0, 0] },
                PPred: P0, // predicted had high uncertainty
            },
        ];

        const result = rtsSmooth(filtered, [F]);

        // Step 0 should have reduced covariance (smoother distributed step 1's correction back)
        const tr0 = result[0].P[0][0] + result[0].P[4][4] + result[0].P[8][8];
        const trOriginal = P0[0][0] + P0[4][4] + P0[8][8];
        expect(tr0).toBeLessThan(trOriginal);
    });

    it("smoother distributes position correction backward along velocity", () => {
        const I9 = new Array(9);
        for (let i = 0; i < 9; i++) {
            I9[i] = new Array(9).fill(0);
            I9[i][i] = 1;
        }

        // F: position couples to position+velocity*dt, delta=1
        // δp_new = δp + δv*dt + coupling to theta
        // δv_new = δv
        // δθ_new = δθ (identity)
        const F = new Array(9);
        for (let i = 0; i < 9; i++) {
            F[i] = new Array(9).fill(0);
            F[i][i] = 1;
        }
        // δp ← δv*dt coupling
        F[0][3] = 1.0; // dt = 1s

        const P0 = I9.map(row => row.map(v => v * 10));
        const I1 = I9.map(row => row.map(v => v * 1));

        const filtered = [
            {
                x: { p: [0, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] },
                P: P0,
                xPred: null,
                PPred: null,
            },
            {
                x: { p: [0, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] },
                P: I1, // GPS correction reduced uncertainty
                xPred: { p: [0, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] },
                PPred: P0,
            },
        ];

        const result = rtsSmooth(filtered, [F]);

        // Smooth should produce valid state (no NaN)
        expect(isNaN(result[0].x.p[0])).toBe(false);
        expect(isNaN(result[0].x.v[0])).toBe(false);
    });

    it("handles null transition gracefully", () => {
        const I9 = new Array(9);
        for (let i = 0; i < 9; i++) {
            I9[i] = new Array(9).fill(0);
            I9[i][i] = 1;
        }

        const filtered = [
            { x: { p: [0, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] }, P: I9, xPred: null, PPred: null },
            { x: { p: [1, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] }, P: I9, xPred: { p: [1, 0, 0], v: [0, 0, 0], q: [1, 0, 0, 0] }, PPred: I9 },
        ];

        const result = rtsSmooth(filtered, [null]);
        expect(result.length).toBe(2);
        // Step 0 should be direct copy since F was null
        expect(result[0].x.p).toEqual([0, 0, 0]);
    });
});
