import { describe, it, expect } from "vitest";
import {
    createRng,
    randn,
    generateCircularTrajectory,
    generateStraightTrajectory,
} from "./synthetic.js";

describe("synthetic — PRNG", () => {
    it("same seed gives same sequence", () => {
        const a = createRng(42);
        const b = createRng(42);
        for (let i = 0; i < 20; i++) {
            expect(a()).toBe(b());
        }
    });

    it("different seeds give different sequences", () => {
        const a = createRng(1);
        const b = createRng(2);
        const vals = new Set();
        for (let i = 0; i < 100; i++) vals.add(a());
        let same = 0;
        for (let i = 0; i < 100; i++) {
            if (vals.has(b())) same++;
        }
        expect(same).toBeLessThan(10);
    });

    it("randn produces approximately normal distribution", () => {
        const rng = createRng(123);
        const samples = [];
        for (let i = 0; i < 1000; i++) {
            samples.push(randn(rng, 5, 2));
        }
        const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
        const std = Math.sqrt(
            samples.reduce((s, v) => s + (v - mean) * (v - mean), 0) / samples.length,
        );
        expect(mean).toBeCloseTo(5, 1);
        expect(std).toBeCloseTo(2, 1);
    });
});

describe("synthetic — circular trajectory", () => {
    it("produces the requested number of samples", () => {
        const traj = generateCircularTrajectory({ durationS: 2, freqHz: 50 });
        expect(traj.length).toBe(100);
    });

    it("starts near origin at t=0", () => {
        const traj = generateCircularTrajectory({ radiusM: 50, speedMs: 15, durationS: 1 });
        const p = traj[0].pNed;
        expect(p.n).toBeCloseTo(0, 1);
        expect(p.e).toBeCloseTo(0, 1);
    });

    it("produces valid quaternions (unit norm)", () => {
        const traj = generateCircularTrajectory({ durationS: 1 });
        for (const pose of traj) {
            const norm = Math.sqrt(
                pose.q[0] ** 2 + pose.q[1] ** 2 + pose.q[2] ** 2 + pose.q[3] ** 2,
            );
            expect(norm).toBeCloseTo(1, 5);
        }
    });

    it("heading increases monotonically (CCW circle)", () => {
        const traj = generateCircularTrajectory({ durationS: 5 });
        for (let i = 1; i < traj.length; i++) {
            const dh = ((traj[i].heading - traj[i - 1].heading + Math.PI * 3) % (Math.PI * 2)) - Math.PI;
            expect(dh).toBeGreaterThan(0);
        }
    });
});

describe("synthetic — straight trajectory", () => {
    it("moves at constant speed in the heading direction", () => {
        const traj = generateStraightTrajectory({
            speedMs: 10,
            headingDeg: 90,
            durationS: 3,
        });

        const p0 = traj[0].pNed;
        const pEnd = traj[traj.length - 1].pNed;

        // Heading 90° = East → e should increase, n should stay ~0
        expect(pEnd.e).toBeGreaterThan(25);
        expect(Math.abs(pEnd.n)).toBeLessThan(1);
        expect(pEnd.e - p0.e).toBeCloseTo(30, 0); // 10 m/s × 3 s
    });
});
