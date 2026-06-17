/**
 * acro1 KML producer — runs the full pipeline and writes a KML + PoseTrack JSON for
 * visual inspection (Google Earth). One-shot producer, not a correctness gate; the gates
 * live in acroFixture.test.js. Skips when the (uncommitted) BFL is absent.
 */
import { describe, it, expect } from "vitest";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { ingestFlightLog, loadFlightLogFromBuffer } from "./flightIngestion.js";
import { estimatePoseTrack } from "./estimatorLoop.js";
import { loadMagCharacterizationModel } from "../mag_model.js";
import { poseTrackToKml } from "./serializers/kmlSerializer.js";
import { poseTrackToJson } from "./serializers/jsonSerializer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DIR = path.resolve(__dirname, "../../../planv5/blackbox/acro1/");
const BFL_PATH = path.join(DIR, "LOG00007.BFL");
const MODEL_PATH = path.join(DIR, "acro1_mag_model.json");

function hasFiles() {
    try { fs.accessSync(BFL_PATH, fs.constants.R_OK); fs.accessSync(MODEL_PATH, fs.constants.R_OK); return true; } catch { return false; }
}

describe("acro1 KML output", () => {
    it("produces a valid KML from the real acro1 log", async () => {
        if (!hasFiles()) { console.warn("SKIP: acro1 files not available"); return; }

        const fl = await loadFlightLogFromBuffer(new Uint8Array(fs.readFileSync(BFL_PATH)));
        const data = ingestFlightLog(fl);

        // Load the mag model only to validate the loader path; heading comes from the FC
        // quaternion. Mag fusion disabled pending WS-B6 (H_mag column indices under
        // 25-state need verification). The QMC5883L calibration is proven good (3–5°
        // heading); the estimator divergence was caused by the quat-prior dedup bug
        // (now fixed). See planv5/18 §36.
        const mr = loadMagCharacterizationModel(JSON.parse(fs.readFileSync(MODEL_PATH, "utf-8")));
        void mr;

        const origin = data.gpsHome || { lat: data.gps[0].lat, lon: data.gps[0].lon, alt: data.gps[0].alt };
        // Default settings — biases observable, normal 15σ GPS gate. The previous
        // "wide gate + frozen biases" workaround was a band-aid for a sign error in the
        // accel-bias→velocity transition Jacobian (eskf.js buildTransition): F[v][b_a]
        // was −R·dt but the strapdown's f = −(accel − b_a) gives ∂v⁺/∂b_a = +R·dt, so
        // the filter corrected b_a the wrong way and the trajectory ran away to ~64 km.
        // With the sign corrected, b_a/b_g converge and the trajectory tracks GPS without
        // freezing biases or disabling the gate.
        const track = estimatePoseTrack({ ...data, mag: [] }, origin, {
            outputHz: 20,
            maxIter: 1,
        });

        // Triads every 8 samples (~0.4 s at 20 Hz) — double the previous density. 2 m axes.
        const kml = poseTrackToKml(track, {
            everyN: 8,
            showTriads: true,
            showPath: true,
            showRawGps: true,
            rawGps: data.gps.map((g) => ({ lat: g.lat, lon: g.lon, alt: g.alt })),
            axisLengthMeters: 2.0,
        });
        const json = poseTrackToJson(track);

        fs.writeFileSync(path.join(DIR, "acro1_track.kml"), kml, "utf-8");
        fs.writeFileSync(path.join(DIR, "acro1_posetrack.json"), json, "utf-8");
        console.log(`  KML ${kml.length} B, JSON ${json.length} B, ${track.samples.length} samples`);

        expect(kml).toContain("kml");
        expect(json).toContain("schemaVersion");
        expect(track.samples.length).toBeGreaterThan(0);
    }, 60000);
});
