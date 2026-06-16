/**
 * acroFixture — real-flight validation suite, driven by a FIXTURE REGISTRY.
 *
 * Each fixture is a (blackbox + mag-model JSON + manifest) triple. The SAME gates run for
 * every fixture, so the suite scales to any validly-configured Betaflight drone, any site,
 * any log rate — add a triple to FIXTURES, no gate-code change. Gates live in acroGates.js
 * (rate-independent by construction). Two layers:
 *   - AUTO gates: the drone's data checking itself (attitude vs FC quaternion, position vs
 *     raw GPS, not-frozen, forward nose≈course) — need no human annotation.
 *   - MANIFEST gates: the pilot's coarse play-by-play as the independent reality anchor
 *     (named maneuvers windowed in video time + loop closure / altitude bounds).
 *
 * BFLs are not committed; a fixture skips gracefully when its log is absent.
 */
import { describe, it, beforeAll, expect } from "vitest";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ingestFlightLog, loadFlightLogFromBuffer, correctMagStream } from "./flightIngestion.js";
import { estimatePoseTrack } from "./estimatorLoop.js";
import { loadMagCharacterizationModel } from "../mag_model.js";
import { llhToNed } from "./geodesy.js";
import {
    gateFinite, gateNotFrozen, gateAttitudeTracksFC, gatePositionTracksGPS, gateForwardCrab,
    runSegmentGate, gateLoopClosure, gateAltDelta, gateMaxClimb,
    quatToR, pitchDeg, windowSamples, videoToTUs,
} from "./acroGates.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---- FIXTURE REGISTRY: add a (bfl, magModel, manifest) triple to scale the suite ----
const FIXTURES = [
    { id: "acro1", dir: "../../../planv5/blackbox/acro1", bfl: "LOG00007.BFL", manifest: "acro1_manifest.json" },
];

function assertGate(skip, gateFn) {
    if (skip) { console.warn("SKIP: fixture BFL not present"); return; }
    const r = gateFn();
    expect(r.pass, r.message).toBe(true);
}

for (const fx of FIXTURES) {
    const dir = path.resolve(__dirname, fx.dir);
    const bflPath = path.join(dir, fx.bfl);
    const manifest = JSON.parse(fs.readFileSync(path.join(dir, fx.manifest), "utf8")); // committed, small
    // Optional per-fixture threshold overrides (undefined → each gate's library default).
    // Lets a noisier-IMU / worse-GPS / gentler-flight drone tune tolerances WITHOUT code
    // changes — the keystone of multi-hardware reuse. Per-segment overrides go in seg.params.
    const T = manifest.thresholds || {};
    const haveBfl = () => { try { fs.accessSync(bflPath, fs.constants.R_OK); return true; } catch { return false; } };

    describe(`acro fixture: ${fx.id}`, () => {
        let ctx = null, modelBoundsOk = false, skip = false;

        beforeAll(async () => {
            if (!haveBfl()) { skip = true; return; }
            const fl = await loadFlightLogFromBuffer(new Uint8Array(fs.readFileSync(bflPath)));
            const d = ingestFlightLog(fl);

            const model = JSON.parse(fs.readFileSync(path.join(dir, manifest.magModel), "utf8"));
            const mr = loadMagCharacterizationModel(model);
            modelBoundsOk = !!mr.model && mr.model.fusion?.qualityBounds?.bounds_ok !== false;

            const origin = d.gpsHome || { lat: d.gps[0].lat, lon: d.gps[0].lon, alt: d.gps[0].alt };

            // Correct mag through the characterization model for 3-axis ESKF fusion (WS-B6)
            const magGauss = mr.model ? correctMagStream(d.mag, mr.model) : [];

            const track = estimatePoseTrack(
                { ...d, mag: magGauss },
                origin,
                {
                    outputHz: 20,
                    magModel: null,  // 3-axis mag fusion corrupts position at 71° inclination (§41)
                },
            );

            ctx = {
                samples: track.samples,
                imu: d.imu,
                baro: d.baro,
                fcQuat: d.quat,
                gpsNed: d.gps.map((g) => ({ tUs: g.tUs, ...llhToNed(g.lat, g.lon, g.alt, origin.lat, origin.lon, origin.alt) })),
                offsetSec: manifest.alignment.offsetSec,
                magRaw: d.mag,  // raw ADC mag samples for cross-validation tests
            };
        }, 120000);

        it("mag model loads and is validly configured (bounds_ok)", () => {
            if (skip) { console.warn("SKIP"); return; }
            expect(modelBoundsOk, "mag model must load with bounds_ok").toBe(true);
        });

        // ---- AUTO gates (no annotation) ----
        it("auto: samples finite", () => assertGate(skip, () => gateFinite(ctx.samples)));
        it("auto: attitude not frozen", () => assertGate(skip, () => gateNotFrozen(ctx.samples, T.notFrozenMinDeg)));
        it("auto: attitude tracks the FC quaternion", () => assertGate(skip, () => gateAttitudeTracksFC(ctx.samples, ctx.fcQuat, T.attitudeMedianMaxDeg)));
        it("auto: position tracks raw GPS", () => assertGate(skip, () => gatePositionTracksGPS(ctx.samples, ctx.gpsNed, T.posDriftMaxM)));
        it("auto: nose tracks course on straight legs", () => assertGate(skip, () => gateForwardCrab(ctx.samples, T.forwardCrabMaxDeg)));

        // ---- MANIFEST segment gates (the play-by-play) ----
        for (const seg of manifest.segments) {
            it(`segment: ${seg.name} [${seg.type}]`, () => assertGate(skip, () => runSegmentGate(ctx, seg)));
        }

        // ---- coarse, pilot-reported ground truth ----
        it("coarse: loop closure", () => assertGate(skip, () => gateLoopClosure(ctx.samples, T.loopMaxM)));
        it("coarse: takeoff↔landing altitude", () => assertGate(skip, () => gateAltDelta(ctx.samples, manifest.coarse.altLowerM, T.altTolM)));
        it("coarse: max climb height", () => assertGate(skip, () => gateMaxClimb(ctx.samples, manifest.coarse.maxClimbM, T.maxClimbTolM)));

        // -------------------------------------------------------------------
        // Pitch validation: cross-check reconstruction attitude against the
        // pilot's camera-based ground truth at key moments (2026-06-15).
        // Tests written after auditing the EXISTING blackbox-log-viewer's
        // attitude display code (flightlog.js computeAttitude, lines 696-714).
        //
        // Viewer approach: NO quaternion conjugation, direct Euler extraction:
        //   pitch = PI/2 - acos(2*(wy-xz)) = -asin(R_body→world[2][0])
        // Our approach: conjugates then extracts standard Euler + rebuilds.
        // BOTH produce identical pitch values for the same input quaternion.
        // The "pitch inversion" during ~0g freefall is a KNOWN DEFECT of the
        // FC's onboard AHRS (low-G gyro-only drift), not an ingestion bug.
        // Cure: 3-axis mag fusion (WS-B6, planv5/21 §4.3, 23 §3).
        // -------------------------------------------------------------------

        /**
         * Test 1 — Climb pitch at video 2:36 (156 s, BB ≈351.3 Mµs).
         * During the powered climb, the FC has a valid gravity reference so
         * its quaternion is correct. The nose should NOT be pitched toward
         * the sky. R[2][0] < −0.1 means nose UP by >~5.7° → FAIL.
         */
        it("pitch: climb nose NOT up (R[2][0] > -0.1 at 156s video)", () => {
            if (skip) { console.warn("SKIP"); return; }
            const tUsTarget = (156 - ctx.offsetSec) * 1e6;  // videoToTUs(156, offsetSec)
            // Find the nearest reconstruction sample
            let best = ctx.samples[0], bestDt = Math.abs(ctx.samples[0].tUs - tUsTarget);
            for (const s of ctx.samples) {
                const dt = Math.abs(s.tUs - tUsTarget);
                if (dt < bestDt) { best = s; bestDt = dt; }
            }
            const R = quatToR(best.q);
            const noseD = R[2][0];  // world-Down component of nose (NED: +Z = down)
            const pitchDegVal = pitchDeg(best.q);
            expect(noseD,
                `nose D-component ${noseD.toFixed(4)} at video 156s ` +
                `(BB ${(tUsTarget/1e6).toFixed(3)}s, pitch ${pitchDegVal.toFixed(1)}°) — ` +
                `nose should NOT be pitched up (need R[2][0] > -0.1)`
            ).toBeGreaterThan(-0.1);
        });

        /**
         * Test 2 — Fall pitch at video 2:45 (165 s, BB ≈360.3 Mµs).
         * The pilot was LOOKING AT THE GROUND (nose DOWN, pitch negative).
         * This is the #1 gaslighting trap: low-G freefall causes the FC
         * quaternion to drift nose-UP (+44° to +77°), and the reconstruction
         * inherits the error. This test ASSERTs nose-down and is EXPECTED
         * TO FAIL until mag fusion (WS-B6) corrects the freefall attitude.
         * A FAIL here = KNOWN DEFECT, tracked as WS-B7.
         *
         * The gate is marked PENDING/RED per planv5/23 §3.1 — DO NOT widen
         * the threshold or delete it. It stays failing until mag fusion lands.
         */
        it("pitch: fall nose DOWN (median pitch < -20° in fall window [163,167]s)", () => {
            if (skip) { console.warn("SKIP"); return; }
            const fallSamples = windowSamples(ctx.samples, 163, 167, ctx.offsetSec);
            if (fallSamples.length === 0) {
                console.warn("SKIP: no samples in fall window");
                return;
            }
            const pitches = fallSamples.map((s) => pitchDeg(s.q));
            pitches.sort((a, b) => a - b);
            const medPitch = pitches[pitches.length >> 1];
            const minPitch = pitches[0];
            const maxPitch = pitches[pitches.length - 1];
            const fcInWindow = windowSamples(ctx.fcQuat, 163, 167, ctx.offsetSec)
                .map((q) => pitchDeg(q.q));
            fcInWindow.sort((a, b) => a - b);
            const fcMedPitch = fcInWindow.length ? fcInWindow[fcInWindow.length >> 1] : NaN;

            // This assertion is expected to FAIL on the pre-mag-fusion estimator.
            // The FC quaternion drifts nose-up during ~0g freefall (planv5/23 §3).
            // The reconstruction inherits this because it anchors on the FC quaternion.
            // Cure: 3-axis mag fusion (WS-B6).
            // Do NOT widen the threshold. Do NOT delete this test.
            const passMsg = `recon median pitch ${medPitch.toFixed(0)}° (range [${minPitch.toFixed(0)}, ${maxPitch.toFixed(0)}]) ` +
                `vs FC median pitch ${Number.isFinite(fcMedPitch) ? fcMedPitch.toFixed(0) : "n/a"}° — ` +
                `need < -20° (nose DOWN). KNOWN FAILURE: low-G AHRS drift. Fix: WS-B6 mag fusion.`;
            expect(medPitch, passMsg).toBeLessThan(-20);
        });

        /**
         * Test 3 — Mag cross-validation at video 2:45 (165 s).
         * At this northern site (inclination +71°), when the drone is
         * physically nose-DOWN, the earth's magnetic field projects strongly
         * onto the body +X axis → magADC[0] should be LARGE (>1000 ADC).
         * This is INDEPENDENT of the FC quaternion — the raw magnetometer
         * confirms the drone WAS nose-down regardless of what the AHRS says.
         * If the reconstruction says nose-UP while magADC[0] is large,
         * the reconstruction is WRONG and the mag is RIGHT.
         */
        it("pitch: mag confirms nose-down at fall (max magADC[0] > 1000 within ±1s of 165s)", () => {
            if (skip) { console.warn("SKIP"); return; }
            const magRaw = ctx.magRaw;
            if (!magRaw || magRaw.length === 0) {
                console.warn("SKIP: no raw mag data available");
                return;
            }
            const tCenter = (165 - ctx.offsetSec) * 1e6;
            const tLo = tCenter - 1e6;  // ±1 second
            const tHi = tCenter + 1e6;
            let maxMagX = -Infinity;
            let countInWindow = 0;
            for (const m of magRaw) {
                if (m.tUs < tLo || m.tUs > tHi) continue;
                countInWindow++;
                // magADC[0] is the raw body-X magnetometer reading (ADC counts)
                const mx = Array.isArray(m.meas) ? m.meas[0] : m.meas;
                if (mx > maxMagX) maxMagX = mx;
            }
            // Also report what the reconstruction says at this time
            const tCenterSample = ctx.samples.reduce((best, s) =>
                Math.abs(s.tUs - tCenter) < Math.abs(best.tUs - tCenter) ? s : best
            );
            const reconPitch = pitchDeg(tCenterSample.q);

            expect(maxMagX,
                `max magADC[0] = ${Number.isFinite(maxMagX) ? maxMagX.toFixed(0) : "n/a"} ADC ` +
                `over ${countInWindow} samples in [164,166]s video. ` +
                `Recon pitch at 165s = ${reconPitch.toFixed(1)}° ` +
                `(${reconPitch < 0 ? "nose-DOWN" : "nose-UP"}). ` +
                `Northern site inclination +71° → earth field projects strongly onto body +X when nose is DOWN. ` +
                `Need max magADC[0] > 1000 to confirm physical nose-down.`
            ).toBeGreaterThan(1000);
        });
    });
}
