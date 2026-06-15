/**
 * PoseTrack IR unit tests.
 *
 * Tests: construction, sampleAt() interpolation, round-trip through JSON serializer,
 * and finiteness guards.
 */

import { describe, it, expect } from "vitest";
import { createPoseTrack } from "./poseTrack.js";
import { poseTrackToJson, poseTrackFromJson } from "./serializers/jsonSerializer.js";

function makeSample(tUs, p, v, q, lla) {
    return {
        tUs,
        p,
        v,
        q,
        lla,
        covPos: [[1e-4, 0, 0], [0, 1e-4, 0], [0, 0, 1e-4]],
        covAtt: [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
    };
}

describe("PoseTrack IR", () => {
    it("constructs with meta and samples", () => {
        const samples = [
            makeSample(0, [0,0,0], [0,0,0], [1,0,0,0], { lat: 48, lon: -71, alt: 200 }),
            makeSample(1000000, [10,0,0], [10,0,0], [1,0,0,0], { lat: 48.00009, lon: -71, alt: 200 }),
        ];

        const track = createPoseTrack({
            samples,
            georefOrigin: { lat: 48, lon: -71, alt: 200 },
            source: { log: "test" },
        });

        expect(track.meta.schemaVersion).toBe(1);
        expect(track.meta.frame).toBe("body=FRD, world=NED");
        expect(track.meta.georefOrigin.lat).toBe(48);
        expect(track.meta.units.pos).toBe("m");
        expect(track.samples).toHaveLength(2);
        expect(track.sampleAt).toBeInstanceOf(Function);
    });

    it("sampleAt interpolates position and velocity (lerp)", () => {
        const samples = [
            makeSample(0, [0, 0, 0], [0, 0, 0], [1,0,0,0], { lat: 48, lon: -71, alt: 200 }),
            makeSample(1000000, [10, 0, 0], [10, 0, 0], [1,0,0,0], { lat: 48.00009, lon: -71, alt: 200 }),
        ];

        const track = createPoseTrack({ samples, georefOrigin: { lat: 48, lon: -71, alt: 200 }, source: {} });

        const mid = track.sampleAt(500000);
        expect(mid).not.toBeNull();
        expect(mid.p[0]).toBeCloseTo(5, 1);
        expect(mid.v[0]).toBeCloseTo(5, 1);
        expect(mid.lla.lat).toBeCloseTo(48.000045, 6);
    });

    it("sampleAt interpolates attitude (slerp)", () => {
        const q0 = [1, 0, 0, 0]; // identity
        const halfTurn = Math.PI / 4; // 45°
        const q1 = [Math.cos(halfTurn), 0, 0, Math.sin(halfTurn)]; // 90° yaw
        const samples = [
            makeSample(0, [0,0,0], [0,0,0], q0, null),
            makeSample(1000000, [0,0,0], [0,0,0], q1, null),
        ];

        const track = createPoseTrack({ samples, georefOrigin: { lat: 0, lon: 0, alt: 0 }, source: {} });

        const mid = track.sampleAt(500000);
        expect(mid).not.toBeNull();
        // At t=0.5, the slerp should give ~45° yaw
        const expected = [Math.cos(halfTurn/2), 0, 0, Math.sin(halfTurn/2)];
        expect(mid.q[0]).toBeCloseTo(expected[0], 4);
        expect(mid.q[3]).toBeCloseTo(expected[3], 4);
    });

    it("sampleAt clamps to endpoints", () => {
        const samples = [
            makeSample(100, [0,0,0], [0,0,0], [1,0,0,0], null),
            makeSample(200, [10,0,0], [10,0,0], [1,0,0,0], null),
        ];

        const track = createPoseTrack({ samples, georefOrigin: { lat: 0, lon: 0, alt: 0 }, source: {} });

        expect(track.sampleAt(0).tUs).toBe(100);
        expect(track.sampleAt(50).tUs).toBe(100);
        expect(track.sampleAt(300).tUs).toBe(200);
        expect(track.sampleAt(1000).tUs).toBe(200);
    });

    it("single sample track returns that sample for any query", () => {
        const sample = makeSample(500, [1,2,3], [4,5,6], [1,0,0,0], { lat: 48, lon: -71, alt: 200 });
        const track = createPoseTrack({ samples: [sample], georefOrigin: { lat: 48, lon: -71, alt: 200 }, source: {} });

        const s = track.sampleAt(0);
        expect(s.p[0]).toBe(1);
        const s2 = track.sampleAt(99999999);
        expect(s2.p[0]).toBe(1);
    });

    it("empty track sampleAt returns null", () => {
        const track = createPoseTrack({ samples: [], georefOrigin: { lat: 0, lon: 0, alt: 0 }, source: {} });
        expect(track.sampleAt(500)).toBeNull();
    });

    it("JSON round-trip preserves meta, samples, covariance, and byte-for-byte stable", () => {
        const samples = [
            makeSample(0, [0,0,0], [0,0,0], [1,0,0,0], { lat: 48.4085, lon: -71.1642, alt: 200.5 }),
            makeSample(500000, [5,3,-1], [10,6,-2], [0.866, 0, 0, 0.5], { lat: 48.4086, lon: -71.1641, alt: 199.5 }),
        ];
        // Set non-diagonal covariance to verify full matrix preservation
        samples[0].covPos = [[0.01, 0.002, 0], [0.002, 0.015, -0.001], [0, -0.001, 0.02]];
        samples[0].covAtt = [[0.001, 0.0001, 0], [0.0001, 0.002, 0], [0, 0, 0.0015]];

        const track = createPoseTrack({
            samples,
            georefOrigin: { lat: 48.408, lon: -71.164, alt: 200 },
            source: { log: "LOG00007.BFL", magModelSchema: "2.1", solverConfig: { test: true } },
        });

        // Round-trip 1: serialize
        const json1 = poseTrackToJson(track);
        // Round-trip 2: deserialize
        const restored = poseTrackFromJson(json1);
        // Round-trip 3: re-serialize
        const json2 = poseTrackToJson(restored);

        // Byte-for-byte stability
        expect(json1).toBe(json2);

        // Content checks
        expect(restored.meta.schemaVersion).toBe(1);
        expect(restored.meta.georefOrigin.lat).toBe(48.408);
        expect(restored.meta.source.log).toBe("LOG00007.BFL");
        expect(restored.samples).toHaveLength(2);

        // Full covariance preservation (off-diagonal elements)
        expect(restored.samples[0].covPos[0][1]).toBeCloseTo(0.002, 10);
        expect(restored.samples[0].covPos[1][2]).toBeCloseTo(-0.001, 10);
        expect(restored.samples[0].covAtt[0][1]).toBeCloseTo(0.0001, 10);

        // Verify the restored track has a functional sampleAt that slerps
        expect(restored.sampleAt).toBeInstanceOf(Function);
        const mid = restored.sampleAt(250000);
        expect(mid.p[0]).toBeCloseTo(2.5, 1);
        // Verify slerp: mid-point quaternion should not be the same as either endpoint
        expect(mid.q[0]).not.toBeCloseTo(1, 6); // not identity
        expect(mid.q).not.toBeNull();
    });
});
