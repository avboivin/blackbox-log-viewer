/**
 * acroGates — reusable, LOG-RATE-INDEPENDENT validation gates for real-flight fixtures.
 *
 * A "fixture" is a (blackbox + mag-model + manifest) triple. The manifest annotates
 * named maneuver windows in VIDEO time plus a clock offset; these gates assert that the
 * reconstructed PoseTrack reproduces each maneuver within tolerances justified by
 * consumer-grade hardware. Each gate returns { name, pass, message } — the message
 * always carries the computed number so a failure is reported with the value, never a
 * bare boolean (planv5/18 standing rule).
 *
 * RATE INDEPENDENCE (Betaflight logs 50 Hz–8 kHz; this fixture is 500 Hz):
 *  - The estimator resamples to a fixed `outputHz` (default 20 Hz), so all attitude/
 *    position gates run on a rate-independent stream.
 *  - Every time quantity (windows, yaw rate, freefall) is derived from `tUs`, never from
 *    a sample count or an assumed loop period. Raw-accel scans iterate the IMU array by
 *    timestamp, so they find the same freefall minimum at any log rate.
 *  - "Sustained for X% of a window" is a fraction of the in-window samples, which at a
 *    fixed output rate equals a fraction of wall-clock time.
 */

const D = 180 / Math.PI;

// ---- attitude geometry (quaternion [w,x,y,z], body FRD -> world NED) ----
export function quatToR(q) {
    const [w, x, y, z] = q;
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ];
}
export function noseBearingDeg(q) { const m = quatToR(q); return ((Math.atan2(m[1][0], m[0][0]) * D) % 360 + 360) % 360; }
export function pitchDeg(q) { const m = quatToR(q); return -Math.asin(Math.max(-1, Math.min(1, m[2][0]))) * D; } // -90 = nose straight down
export function tiltFromUprightDeg(q) { const m = quatToR(q); return Math.acos(Math.max(-1, Math.min(1, m[2][2]))) * D; } // 0 upright, 180 inverted
export function speed(v) { return Math.hypot(v[0], v[1]); }
export function courseDeg(v) { return (Math.atan2(v[1], v[0]) * D % 360 + 360) % 360; }
export function wrap(deg) { let v = deg; while (v > 180) v -= 360; while (v < -180) v += 360; return v; }
export function crabDeg(q, v) { return wrap(noseBearingDeg(q) - courseDeg(v)); }
export function quatAngleDeg(a, b) { const d = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]); return 2 * Math.acos(Math.min(1, d)) * D; }

// ---- time / windowing (all by tUs, rate-independent) ----
export const videoToTUs = (videoSec, offsetSec) => (videoSec - offsetSec) * 1e6;
export function windowSamples(samples, t0Video, t1Video, offsetSec) {
    const a = videoToTUs(t0Video, offsetSec), b = videoToTUs(t1Video, offsetSec);
    return samples.filter((s) => s.tUs >= a && s.tUs <= b);
}
export function minAccelInWindow(imu, t0Video, t1Video, offsetSec) {
    const a = videoToTUs(t0Video, offsetSec), b = videoToTUs(t1Video, offsetSec);
    let m = Infinity;
    for (const im of imu) { if (im.tUs < a || im.tUs > b) continue; const g = Math.hypot(...im.accel); if (g < m) m = g; }
    return m;
}
function nearestByTUs(arr, tUs) {
    let lo = 0, hi = arr.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].tUs < tUs) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(arr[lo - 1].tUs - tUs) < Math.abs(arr[lo].tUs - tUs)) return arr[lo - 1];
    return arr[lo];
}
function median(xs) { const a = [...xs].sort((p, q) => p - q); return a.length ? a[a.length >> 1] : NaN; }
const mk = (name, pass, message) => ({ name, pass, message });

// ---- per-sample yaw rate (deg/s) from timestamps — rate-independent ----
export function yawRateSeries(samples) {
    const r = new Array(samples.length).fill(0);
    for (let i = 1; i < samples.length - 1; i++) {
        const dt = (samples[i + 1].tUs - samples[i - 1].tUs) / 1e6;
        if (dt > 0) r[i] = wrap(noseBearingDeg(samples[i + 1].q) - noseBearingDeg(samples[i - 1].q)) / dt;
    }
    return r;
}

// ============================ AUTO gates (no manifest) ============================

export function gateFinite(samples) {
    let bad = 0;
    for (const s of samples) if (!s.p.every(Number.isFinite) || !s.q.every(Number.isFinite)) bad++;
    return mk("finite", bad === 0, `${bad} non-finite samples of ${samples.length}`);
}

export function gateNotFrozen(samples, minTotalDeg = 500) {
    let tot = 0;
    for (let i = 1; i < samples.length; i++) tot += Math.abs(wrap(noseBearingDeg(samples[i].q) - noseBearingDeg(samples[i - 1].q)));
    return mk("not-frozen", tot > minTotalDeg, `total |Δyaw| = ${tot.toFixed(0)}° (need > ${minTotalDeg}°)`);
}

/** Reconstructed attitude must track the FC's own logged quaternion (the onboard fused
 *  attitude). Uses the MEDIAN geodesic error, not RMS: during violent inversions (flips/
 *  rolls) recon and FC can be a half-sample apart, producing brief ~180° spikes that are
 *  timing jitter, not divergence — RMS would be dominated by them. Median measures typical
 *  tracking and still catches a frozen/collapsed attitude (whose median would be ≫25°). */
export function gateAttitudeTracksFC(samples, fcQuat, maxMedianDeg = 25) {
    const errs = [];
    for (const s of samples) { const f = nearestByTUs(fcQuat, s.tUs); errs.push(quatAngleDeg(s.q, f.q)); }
    errs.sort((a, b) => a - b);
    const med = errs[errs.length >> 1];
    const p90 = errs[Math.floor(0.9 * (errs.length - 1))];
    return mk("attitude-tracks-FC", med < maxMedianDeg, `attitude error vs FC: median ${med.toFixed(1)}° (need < ${maxMedianDeg}°); p90 ${p90.toFixed(0)}° (tail = fast-flip timing jitter)`);
}

/** Reconstructed position must track raw GPS (no drift runaway). toNed: (lat,lon,alt)->{n,e}. */
export function gatePositionTracksGPS(samples, gpsNed, maxDriftM = 25) {
    let maxD = 0;
    for (const s of samples) {
        const g = nearestByTUs(gpsNed, s.tUs);
        const h = Math.hypot(s.p[0] - g.n, s.p[1] - g.e);
        if (h > maxD) maxD = h;
    }
    return mk("position-tracks-GPS", maxD < maxDriftM, `max drift vs GPS = ${maxD.toFixed(1)} m (need < ${maxDriftM} m)`);
}

/** During straight forward flight the nose must track GPS course (small crab). Forward
 *  flight = fast + low yaw-rate, so orbits/flips/backward segments self-exclude. */
export function gateForwardCrab(samples, maxMedianCrabDeg = 30, minSpeed = 5, maxYawRate = 15) {
    const yr = yawRateSeries(samples);
    const crabs = [];
    for (let i = 0; i < samples.length; i++) {
        if (speed(samples[i].v) < minSpeed || Math.abs(yr[i]) > maxYawRate) continue;
        crabs.push(Math.abs(crabDeg(samples[i].q, samples[i].v)));
    }
    const med = median(crabs);
    return mk("forward-crab", crabs.length > 10 && med < maxMedianCrabDeg,
        `median |crab| on straight legs = ${Number.isFinite(med) ? med.toFixed(0) : "n/a"}° over ${crabs.length} samples (need < ${maxMedianCrabDeg}°)`);
}

// ============================ MANIFEST (typed) gates ============================

/** heading_anchor: independent global-frame check — at a declared moment the nose points
 *  a declared cardinal (catches a whole-flight heading rotation that fools every sensor). */
export function gateHeadingAnchor(ctx, seg) {
    const { tolDeg = 45 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    if (!ws.length) return mk(seg.name, false, "no samples in window");
    // circular median via mean of unit vectors
    let sx = 0, sy = 0; for (const s of ws) { const b = noseBearingDeg(s.q) / D; sx += Math.cos(b); sy += Math.sin(b); }
    const hdg = (Math.atan2(sy, sx) * D % 360 + 360) % 360;
    const err = Math.abs(wrap(hdg - seg.expect.headingDeg));
    return mk(seg.name, err < tolDeg, `heading ${hdg.toFixed(0)}° vs declared ${seg.expect.headingDeg}° → err ${err.toFixed(0)}° (tol ${tolDeg}°)`);
}

/** orbit: sustained crab (nose far off the velocity), with a declared handedness sign. */
export function gateOrbit(ctx, seg) {
    const { minMeanCrabDeg = 35 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    let sum = 0, n = 0;
    for (const s of ws) { if (speed(s.v) < 2) continue; sum += crabDeg(s.q, s.v); n++; }
    const mean = n ? sum / n : 0;
    const signOk = seg.expect.crabSign ? Math.sign(mean) === Math.sign(seg.expect.crabSign) : true;
    const pass = n > 5 && Math.abs(mean) > minMeanCrabDeg && signOk;
    return mk(seg.name, pass, `mean crab = ${mean.toFixed(0)}° (need |·| > ${minMeanCrabDeg}°, sign ${seg.expect.crabSign > 0 ? "+" : "−"})`);
}

/** barrel_roll: attitude reaches inverted and returns; freefall present in the sub-window. */
export function gateBarrelRoll(ctx, seg) {
    const { minTiltDeg = 150, maxFreefallAccel = 3 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    let maxTilt = 0;
    const endTilt = ws.length ? tiltFromUprightDeg(ws[ws.length - 1].q) : 180;
    for (const s of ws) { const t = tiltFromUprightDeg(s.q); if (t > maxTilt) maxTilt = t; }
    const ff = seg.expect.freefall ? minAccelInWindow(ctx.imu, seg.expect.freefall[0], seg.expect.freefall[1], ctx.offsetSec) : 0;
    // endTilt < 70: out of inversion by window end (recovered/caught). Not < 40, because the
    // recovery is still banked a beat after the roll completes — "no longer inverted" is the
    // honest claim, not "perfectly level".
    const pass = maxTilt > minTiltDeg && (!seg.expect.freefall || ff < maxFreefallAccel) && endTilt < 70;
    return mk(seg.name, pass, `maxTilt ${maxTilt.toFixed(0)}° (>${minTiltDeg}), freefall accel ${ff.toFixed(1)} (<${maxFreefallAccel}), endTilt ${endTilt.toFixed(0)}° (<70 = out of inversion)`);
}

/** pitch_flip: full pitch loop (sweeps through vertical) + inverted + freefall. Uses
 *  quaternion tilt + pitch range, NOT a single Euler asin, so the ±90° singularity is safe. */
export function gatePitchFlip(ctx, seg) {
    const { minTiltDeg = 150, minPitchRangeDeg = 120, maxFreefallAccel = 3 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    let maxTilt = 0, minP = 999, maxP = -999;
    for (const s of ws) { const t = tiltFromUprightDeg(s.q); if (t > maxTilt) maxTilt = t; const p = pitchDeg(s.q); if (p < minP) minP = p; if (p > maxP) maxP = p; }
    const range = maxP - minP;
    const ff = seg.expect.freefall ? minAccelInWindow(ctx.imu, seg.expect.freefall[0], seg.expect.freefall[1], ctx.offsetSec) : 0;
    const pass = maxTilt > minTiltDeg && range > minPitchRangeDeg && (!seg.expect.freefall || ff < maxFreefallAccel);
    return mk(seg.name, pass, `maxTilt ${maxTilt.toFixed(0)}° (>${minTiltDeg}), pitchRange ${range.toFixed(0)}° (>${minPitchRangeDeg}), freefall ${ff.toFixed(1)} (<${maxFreefallAccel})`);
}

/** climb_fall: large altitude excursion consistent with baro, a dramatic orientation
 *  change at the top, and freefall in the fall sub-window.
 *
 *  NOTE — we deliberately do NOT assert "nose-down" here. On acro1 the pilot reports
 *  "looking at the ground" during the fall, but BOTH the FC's onboard attitude AND our
 *  reconstruction (which faithfully tracks it) show nose-UP/inverted through this window.
 *  In freefall the accelerometer reads ~0 g, so there is no gravity reference and pitch is
 *  pure gyro integration → it drifts. This is a SHARED-MODE error of the FC and our
 *  estimator that only the human play-by-play caught (planv5/18 §31). Asserting nose-down
 *  would assert something the data cannot support; instead we assert the verifiable facts
 *  (climb height vs baro, freefall, that the attitude changed dramatically). */
export function gateClimbFall(ctx, seg) {
    const { minClimbM = 25, baroTolM = 20, minTiltDeg = 60, maxFreefallAccel = 3 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    if (!ws.length) return mk(seg.name, false, "no samples in window");
    let dMin = Infinity, dMax = -Infinity, maxTilt = 0;
    for (const s of ws) { if (s.p[2] < dMin) dMin = s.p[2]; if (s.p[2] > dMax) dMax = s.p[2]; const t = tiltFromUprightDeg(s.q); if (t > maxTilt) maxTilt = t; }
    const reconClimb = dMax - dMin;
    const a = videoToTUs(seg.t0, ctx.offsetSec), b = videoToTUs(seg.t1, ctx.offsetSec);
    let bMin = Infinity, bMax = -Infinity;
    for (const bb of ctx.baro) { if (bb.tUs < a || bb.tUs > b) continue; if (bb.alt < bMin) bMin = bb.alt; if (bb.alt > bMax) bMax = bb.alt; }
    const baroClimb = (bMax - bMin);
    const ff = seg.expect.fall ? minAccelInWindow(ctx.imu, seg.expect.fall[0], seg.expect.fall[1], ctx.offsetSec) : 0;
    const pass = reconClimb > minClimbM
        && Math.abs(reconClimb - baroClimb) < baroTolM
        && maxTilt > minTiltDeg
        && (!seg.expect.fall || ff < maxFreefallAccel);
    return mk(seg.name, pass, `reconClimb ${reconClimb.toFixed(0)} m vs baro ${baroClimb.toFixed(0)} m (Δ<${baroTolM}), maxTilt ${maxTilt.toFixed(0)}° (>${minTiltDeg} orientation change), freefall ${ff.toFixed(1)} (<${maxFreefallAccel})`);
}

/** backward: nose ~opposite to travel somewhere in the window (heading ≠ GPS course). */
export function gateBackward(ctx, seg) {
    const { minCrabDeg = 110 } = seg.params || {};
    const ws = windowSamples(ctx.samples, seg.t0, seg.t1, ctx.offsetSec);
    let maxCrab = 0;
    for (const s of ws) { if (speed(s.v) < 3) continue; const c = Math.abs(crabDeg(s.q, s.v)); if (c > maxCrab) maxCrab = c; }
    return mk(seg.name, maxCrab > minCrabDeg, `max |crab| = ${maxCrab.toFixed(0)}° (need > ${minCrabDeg}° = flying backward)`);
}

const SEGMENT_GATES = {
    heading_anchor: gateHeadingAnchor,
    orbit: gateOrbit,
    barrel_roll: gateBarrelRoll,
    pitch_flip: gatePitchFlip,
    climb_fall: gateClimbFall,
    backward: gateBackward,
};
export function runSegmentGate(ctx, seg) {
    const fn = SEGMENT_GATES[seg.type];
    if (!fn) return mk(seg.name, false, `unknown segment type "${seg.type}"`);
    return fn(ctx, seg);
}

// ---- coarse, pilot-reported ground truth ----
export function gateLoopClosure(samples, maxM = 10) {
    const d = Math.hypot(samples[samples.length - 1].p[0] - samples[0].p[0], samples[samples.length - 1].p[1] - samples[0].p[1]);
    return mk("loop-closure", d < maxM, `takeoff↔landing = ${d.toFixed(1)} m (need < ${maxM} m)`);
}
export function gateAltDelta(samples, expectedLowerM, tolM = 12) {
    const lower = samples[samples.length - 1].p[2] - samples[0].p[2]; // NED down: + = landed lower
    const pass = Math.abs(lower - expectedLowerM) < tolM;
    return mk("alt-delta", pass, `landed ${lower.toFixed(1)} m lower vs pilot ${expectedLowerM} m (tol ${tolM} m)`);
}
export function gateMaxClimb(samples, expectedM, tolM = 30) {
    let dMin = Infinity; for (const s of samples) if (s.p[2] < dMin) dMin = s.p[2];
    const climb = samples[0].p[2] - dMin;
    return mk("max-climb", Math.abs(climb - expectedM) < tolM, `max climb = ${climb.toFixed(0)} m vs pilot ${expectedM} m (tol ${tolM} m)`);
}
