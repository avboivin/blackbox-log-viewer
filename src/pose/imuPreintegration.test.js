import { describe, it, expect } from "vitest";
import { createPreintegrator, preintegrateStep, predictState } from "./imuPreintegration.js";

describe("imuPreintegration", () => {
    it("accumulates zero delta when drone is hovering (no motion)", () => {
        const preint = createPreintegrator();

        // Hover: gyro zero, accel reads +g on Z (level drone)
        const omega = [0, 0, 0];
        const accel = [0, 0, 9.80665];
        const dt = 0.01;

        // First call stores sample, subsequent calls integrate.
        // 51 iterations = 50 integration steps × 0.01 = 0.5s
        for (let i = 0; i < 51; i++) {
            preintegrateStep(preint, omega, accel, dt);
        }

        expect(preint.dtSum).toBeCloseTo(0.5, 2);

        // dR should stay identity (no rotation)
        const dR = preint.dR;
        expect(dR[0][0]).toBeCloseTo(1, 5);
        expect(dR[0][1]).toBeCloseTo(0, 5);
        expect(dR[2][2]).toBeCloseTo(1, 5);

        // dv and dp should be near zero (specific force cancels gravity in predictState)
        // The raw accel is [0,0,9.81] → specific force = [0,0,-9.81]
        // dv = [0, 0, -9.81 * 0.5] = [0, 0, -4.905]
        // After predictState adds g_world * dtSum = +4.905, result is zero velocity change
        const p_i = [0, 0, 0];
        const v_i = [0, 0, 0];
        const R_i = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const pred = predictState(preint, p_i, v_i, R_i);

        // Velocity should stay zero (gravity cancels)
        expect(pred.v_j[2]).toBeCloseTo(0, 4);
        // Position should stay zero
        expect(pred.p_j[2]).toBeCloseTo(0, 3);
    });

    it("predicts correct forward motion for horizontal acceleration", () => {
        const preint = createPreintegrator();

        // Drone accelerates forward: gyro zero, accel has horizontal component + gravity
        const omega = [0, 0, 0];
        const accel = [1.0, 0, 9.80665];  // 1 m/s² forward in body X + gravity on Z
        const dt = 0.01;

        for (let i = 0; i < 51; i++) {
            preintegrateStep(preint, omega, accel, dt);
        }

        expect(preint.dtSum).toBeCloseTo(0.5, 2);

        const p_i = [0, 0, 0];
        const v_i = [0, 0, 0];
        const R_i = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const pred = predictState(preint, p_i, v_i, R_i);

        // dv = dR_prev · acMid · dt accumulated. With accel = [1,0,9.81]:
        // specific force = [-1, 0, -9.81]
        // dv_X ≈ -1 * 0.5 = -0.5
        // predictState: v_j = R_i · dv + g·dtSum
        // v_j_X = -0.5 (no gravity on X)
        // v_j_Z = -9.81*0.5 + 9.81*0.5 = 0
        expect(pred.v_j[0]).toBeCloseTo(-0.5, 1);
        expect(pred.v_j[2]).toBeCloseTo(0, 4);
    });

    it("covariance grows with accumulation", () => {
        const preint = createPreintegrator();
        const initTrace = preint.cov[0][0] + preint.cov[4][4] + preint.cov[8][8];

        const omega = [0, 0, 0];
        const accel = [0, 0, 9.80665];
        const dt = 0.01;

        for (let i = 0; i < 100; i++) {
            preintegrateStep(preint, omega, accel, dt);
        }

        const finalTrace = preint.cov[0][0] + preint.cov[4][4] + preint.cov[8][8];
        expect(finalTrace).toBeGreaterThan(initTrace);
    });

    it("rotation integration works for yaw turn", () => {
        const preint = createPreintegrator();

        // Yaw at 1 rad/s for 0.5s → 0.5 rad yaw change
        const omega = [0, 0, 1.0];
        const accel = [0, 0, 9.80665];
        const dt = 0.01;

        for (let i = 0; i < 51; i++) {
            preintegrateStep(preint, omega, accel, dt);
        }

        expect(preint.dtSum).toBeCloseTo(0.5, 2);

        // dR should be a rotation about Z by ~0.5 rad
        const dR = preint.dR;
        // R = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
        expect(dR[0][0]).toBeCloseTo(Math.cos(0.5), 1);
        expect(dR[0][1]).toBeCloseTo(-Math.sin(0.5), 1);
        expect(dR[1][0]).toBeCloseTo(Math.sin(0.5), 1);
        expect(dR[1][1]).toBeCloseTo(Math.cos(0.5), 1);
        expect(dR[2][2]).toBeCloseTo(1, 5);
    });
});
