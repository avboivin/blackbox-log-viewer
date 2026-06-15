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
                    magModel: mr.model?.fusion || null,
                },
            );

            ctx = {
                samples: track.samples,
                imu: d.imu,
                baro: d.baro,
                fcQuat: d.quat,
                gpsNed: d.gps.map((g) => ({ tUs: g.tUs, ...llhToNed(g.lat, g.lon, g.alt, origin.lat, origin.lon, origin.alt) })),
                offsetSec: manifest.alignment.offsetSec,
            };
        });

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
    });
}
