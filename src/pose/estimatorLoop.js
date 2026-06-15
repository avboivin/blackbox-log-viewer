/**
 * Estimator loop — orchestrates forward ESKF pass, RTS backward smoothing,
 * and Gauss-Newton iteration to produce maximum-a-posteriori pose estimates.
 *
 * Schedules a keyframe at configurable rate. Between keyframes the forward
 * ESKF integrates IMU per-step. At each keyframe GPS/baro/quaternion updates
 * are applied. The backward RTS smoother then distributes corrections.
 */

import { createEskf, eskfPredict, eskfUpdate } from "./eskf.js";
import { createGpsPositionFactor, createGpsPositionFactorWithLatency, createGpsVelocityFactor, createBaroFactor, createQuaternionPrior, createMagFactor, createDeclinationFactor } from "./measurements.js";
import { rtsSmooth } from "./rtsSmoother.js";
import { llhToNed, nedToLlh } from "./geodesy.js";
import { createPoseTrack } from "./poseTrack.js";

/**
 * Run the estimation pipeline over pre-parsed sensor data.
 *
 * This is the flat backward-compatible output. New consumers should use
 * estimatePoseTrack() which returns the full PoseTrack IR.
 *
 * @param {object} data
 * @param {object} origin - {lat, lon, alt}
 * @param {object} [opts]
 * @returns {Array<{tMs:number, lat:number, lon:number, altMsl:number, q:[4], vNed:[3], sigmaPos:number, sigmaAtt:number}>}
 */
export function estimatePoses(data, origin, opts = {}) {
    const track = estimatePoseTrack(data, origin, opts);
    const t0Us = track.samples.length > 0 ? track.samples[0].tUs : 0;
    return track.samples.map((s) => ({
        tMs: (s.tUs - t0Us) / 1000,
        lat: s.lla ? s.lla.lat : origin.lat,
        lon: s.lla ? s.lla.lon : origin.lon,
        altMsl: s.lla ? s.lla.alt : origin.alt,
        q: s.q,
        vNed: s.v,
        sigmaPos: Math.sqrt(Math.max(0, (s.covPos[0][0] + s.covPos[1][1] + s.covPos[2][2]) / 3)),
        sigmaAtt: Math.sqrt(Math.max(0, (s.covAtt[0][0] + s.covAtt[1][1] + s.covAtt[2][2]) / 3)) * (180 / Math.PI),
    }));
}

// ---------------------------------------------------------------------------
// Internal: shared estimation core
// ---------------------------------------------------------------------------

/**
 * Run the core estimation loop and return smoothed states with full covariance.
 * Called by estimatePoseTrack. Returns { smoothed, t0Us, lat0, lon0, alt0 }.
 */
function _runEstimation(data, origin, opts = {}) {
    const {
        outputHz = 20,
        gpsPosSigma = 2.5,
        gpsVelSigma = 0.5,
        baroSigma = 1.0,
        attSigma = 0.1,
        maxIter = 3,
        magSigma = 0.05,
        declSigma = 0.34,
        magModel = null,
        useKI = false,
        useTau = false,
        useDcs = false,
        current = null,
        procSigmaAcc = 0.35,   // AP EKF3 default: accelerometer process noise 1σ (m/s²)
        procSigmaGyro = 0.015,  // AP EKF3 default: gyroscope process noise 1σ (rad/s)
        // GPS innovation gates (σ-multiples). Principled 5σ per ArduPilot EKF3.
        // The old 15σ band-aid was necessary without bias states (b_a/b_g) because
        // uncompensated IMU bias caused legitimate IMU divergence between GPS fixes
        // that a tight gate misread as outliers. With unconditional b_a/b_g states
        // (Task A, planv5 Q3), bias is estimated explicitly — IMU prediction stays
        // within a 5σ gate, and the 15σ gate becomes the loosening, not the cure.
        gpsPosGate = 5.0,
        gpsVelGate = 5.0,
    } = opts;

    const { imu, gps, baro, quat, mag } = data;
    if (!imu || imu.length === 0) return { smoothed: [], t0Us: 0, lat0: 0, lon0: 0, alt0: 0 };

    const { lat: lat0, lon: lon0, alt: alt0 } = origin;
    const t0Us = imu[0].tUs;

    // ---- Initial state from first GPS ----
    let p0 = [0, 0, 0];
    let v0 = [0, 0, 0];
    let q0 = [1, 0, 0, 0];

    if (gps.length > 0) {
        const g0 = llhToNed(gps[0].lat, gps[0].lon, gps[0].alt, lat0, lon0, alt0);
        p0 = [g0.n, g0.e, g0.d];
        if (gps[0].velNed) v0 = [...gps[0].velNed];
        if (gps.length >= 2) {
            const dtS = (gps[1].tUs - gps[0].tUs) / 1e6;
            if (dtS > 0 && !gps[0].velNed) {
                const g1 = llhToNed(gps[1].lat, gps[1].lon, gps[1].alt, lat0, lon0, alt0);
                v0 = [(g1.n - g0.n) / dtS, (g1.e - g0.e) / dtS, (g1.d - g0.d) / dtS];
            }
        }
    }
    if (quat.length > 0) q0 = quat[0].q;

    // ---- Static-window bias initialization (Q1) ----
    // b_g: fully observable from static hold — compute mean gyro, tight prior
    // b_a: NOT observable from single static orientation — zero init, moderate prior
    let bg0 = [0, 0, 0];
    let sigmaBgInit = 0.05;  // default: loose prior (no static window)
    const staticWindowUs = 5e6;  // first 5 seconds
    const staticImu = imu.filter((x) => x.tUs - imu[0].tUs <= staticWindowUs);
    if (staticImu.length > 100) {
        // Static window present: b_g from mean gyro, TIGHT prior
        const sumG = [0, 0, 0];
        for (const x of staticImu) {
            sumG[0] += x.gyro[0]; sumG[1] += x.gyro[1]; sumG[2] += x.gyro[2];
        }
        const n = staticImu.length;
        bg0 = [sumG[0]/n, sumG[1]/n, sumG[2]/n];
        // After 5s of averaging: σ ≈ gyro_noise/√(n·dt) ≈ 0.015/√(2500·0.001) ≈ 0.01 rad/s
        sigmaBgInit = 0.01;
    }
    const ba0 = [0, 0, 0];
    const sigmaBaInit = 0.5;  // moderate prior, refined in flight (Q1)

    // Baro offset from first GPS altitude
    let baroOffset = 0;
    if (baro.length > 0 && gps.length > 0) {
        const baroAltAtGps0 = findBaroAtTime(baro, gps[0].tUs);
        if (baroAltAtGps0 !== null) {
            baroOffset = gps[0].alt - baroAltAtGps0;
        }
    }

    // ---- Build keyframe schedule ----
    const outputIntervalUs = 1e6 / outputHz;
    const hasMag = magModel && magModel.earthFieldNedGauss && mag && mag.length > 0;
    const useMag = hasMag && magModel.qualityBounds?.bounds_ok !== false;
    let poses = [];

    // Mag noise from model or default
    const magMeasSigma = useMag && magModel.magNoiseGauss?.sigma != null
        ? magModel.magNoiseGauss.sigma
        : magSigma;

    for (let iter = 0; iter < maxIter; iter++) {
        const eskfOpts = { p0, v0, q0, sigmaPos: 5, sigmaVel: 2, sigmaAtt: 0.2,
            ba0, bg0, sigmaBa: sigmaBaInit, sigmaBg: sigmaBgInit,
            procSigmaAcc, procSigmaGyro };
        if (useMag) {
            const me = magModel.earthFieldNedGauss;
            eskfOpts.mEarth0 = [me.n, me.e, me.d];
            eskfOpts.mBody0 = [0, 0, 0];
        }
        if (useTau) eskfOpts.tauGps0 = 0.05; // initial latency estimate ~50 ms
        if (useKI) eskfOpts.kI0 = [0, 0, 0];
        const eskf = createEskf(eskfOpts);
        const steps = [];
        let gpsIdx = 0;
        let baroIdx = 0;
        let quatIdx = 0;
        let magIdx = 0;

        let imuIdx = 0;
        let nextKfUs = imu[0].tUs + outputIntervalUs;
        let F_acc = buildIdentityF(eskf.dim);

        while (imuIdx < imu.length) {
            const nowUs = imu[imuIdx].tUs;

            // Predict one IMU step
            const dtUs = imuIdx < imu.length - 1 ? imu[imuIdx + 1].tUs - imu[imuIdx].tUs : 0;
            let F_step = null;
            if (dtUs > 0) {
                const result = eskfPredict(eskf, imu[imuIdx].gyro, imu[imuIdx].accel, dtUs / 1e6);
                F_step = result.F;
            }

            // Accumulate per-step F: F_acc ← F_step · F_acc
            if (F_step) {
                F_acc = matMulFn(F_step, F_acc);
            }

            // ---- Updates at keyframe boundary ----
            if (nowUs >= nextKfUs || imuIdx === imu.length - 1) {
                const xPred = { p: [...eskf.p], v: [...eskf.v], q: [...eskf.q], ba: [...eskf.ba], bg: [...eskf.bg], tUs: nowUs };
                if (useMag) {
                    xPred.mEarth = eskf.mEarth ? [...eskf.mEarth] : undefined;
                    xPred.mBody = eskf.mBody ? [...eskf.mBody] : undefined;
                }
                if (useTau) xPred.tauGps = eskf.tauGps;
                if (useKI) xPred.kI = eskf.kI ? [...eskf.kI] : undefined;
                const PPred = eskf.P.map((r) => [...r]);

                let hasUpdate = false;

                // Find nearest current for mag k_I term
                const findCurrent = (tUs) => {
                    if (!current || !current.length) return 0;
                    let best = current[0];
                    let bestDt = Math.abs(current[0].tUs - tUs);
                    for (let ci = 1; ci < current.length; ci++) {
                        const dt = Math.abs(current[ci].tUs - tUs);
                        if (dt < bestDt) { bestDt = dt; best = current[ci]; }
                    }
                    return best.amps;
                };
                const currAmps = useKI ? findCurrent(nowUs) : 0;

                // GPS update if fix is available near this time
                while (gpsIdx < gps.length && gps[gpsIdx].tUs <= nextKfUs + outputIntervalUs * 0.5) {
                    const gpsF = gps[gpsIdx];
                    const gNed = llhToNed(gpsF.lat, gpsF.lon, gpsF.alt, lat0, lon0, alt0);
                    const fP = useTau
                        ? createGpsPositionFactorWithLatency({ n: gNed.n, e: gNed.e, d: gNed.d }, gpsPosSigma)
                        : createGpsPositionFactor({ n: gNed.n, e: gNed.e, d: gNed.d }, gpsPosSigma);
                    const gpsRobustOpts = useDcs ? { dcs: true, dcsPhi: 1.0 } : {};
                    if (eskfUpdate(eskf, fP, { n: gNed.n, e: gNed.e, d: gNed.d }, gpsPosGate, gpsRobustOpts)) hasUpdate = true;
                    if (gpsF.velNed) {
                        const fV = createGpsVelocityFactor(
                            { n: gpsF.velNed[0], e: gpsF.velNed[1], d: gpsF.velNed[2] },
                            gpsVelSigma,
                        );
                        if (eskfUpdate(eskf, fV, { n: gpsF.velNed[0], e: gpsF.velNed[1], d: gpsF.velNed[2] }, gpsVelGate, gpsRobustOpts))
                            hasUpdate = true;
                    }
                    gpsIdx++;
                }

                // Baro update — only the LAST sample before this keyframe.
                // Baro is logged at I-frame rate (500 Hz), but the value changes only at
                // ~Hz rates. Fusing 25 near-identical corrections per keyframe drives P_D
                // to near-zero (measurement over-counting) → the filter becomes overconfident
                // and rejects the baro when altitude actually changes. One update per keyframe
                // (20 Hz) is the correct bandwidth for a barometric altimeter.
                // (planv5/18 §35 — baro over-counting, 2026-06-15)
                {
                    let lastBaro = null;
                    while (baroIdx < baro.length && baro[baroIdx].tUs <= nextKfUs) {
                        lastBaro = baro[baroIdx];
                        baroIdx++;
                    }
                    if (lastBaro) {
                        const fB = createBaroFactor(lastBaro.alt, baroOffset, baroSigma);
                        if (eskfUpdate(eskf, fB, lastBaro.alt)) hasUpdate = true;
                    }
                }

                // Quaternion prior — fused WITHOUT chi-square gating (gate=Infinity).
                // Unlike GPS/mag (outlier-prone exteroceptive sensors), the logged
                // imuQuaternion is the FC's own fused attitude — the trusted anchor we
                // are reconstructing. During aggressive flight the gyro-propagated
                // attitude drifts >30° from it between corrections, which saturates a
                // gate=3 innovation test → the prior is rejected → yaw free-runs on the
                // drifting (un-bias-corrected) gyro and the heading detaches from truth.
                // Gating the reference against free-running dead-reckoning is backwards;
                // when they disagree the FC quaternion is the more trustworthy of the
                // two. See 18 §29 (real-flight: ungating restores heading tracking to ±25°,
                // reproduces the 180° yaw reversals; gated, yaw froze).
                //
                // Only the LAST quaternion sample before the keyframe is fused. Like baro,
                // the quaternion is logged at I-frame rate (500 Hz) — applying 25 near-
                // identical corrections without interleaving gyro steps drives the attitude
                // covariance P_θ to near-zero (measurement over-counting). The filter then
                // trusts its own (drifting) integration over the FC anchor, and attitude
                // diverges. One prior per keyframe is the correct bandwidth.
                // (planv5/18 §35 — quaternion-prior over-counting, 2026-06-15)
                {
                    let lastQuat = null;
                    while (quatIdx < quat.length && quat[quatIdx].tUs <= nextKfUs) {
                        lastQuat = quat[quatIdx];
                        quatIdx++;
                    }
                    if (lastQuat) {
                        const fQ = createQuaternionPrior(lastQuat.q, attSigma);
                        if (eskfUpdate(eskf, fQ, lastQuat.q, Infinity)) hasUpdate = true;
                    }
                }

                // 3-axis mag update (gate 3.0 per 09 §1)
                if (useMag) {
                    while (magIdx < mag.length && mag[magIdx].tUs <= nextKfUs) {
                        const fM = createMagFactor(mag[magIdx].meas, magMeasSigma, currAmps);
                        if (eskfUpdate(eskf, fM, mag[magIdx].meas, 3.0)) hasUpdate = true;
                        magIdx++;
                    }

                    // Declination pseudo-measurement (once per keyframe if mag updates were applied)
                    if (hasUpdate && magModel.earthFieldNedGauss) {
                        const me = eskf.mEarth;
                        if (me) {
                            const decl = Math.atan2(magModel.earthFieldNedGauss.e, magModel.earthFieldNedGauss.n);
                            const fD = createDeclinationFactor(decl, declSigma);
                            eskfUpdate(eskf, fD, decl);
                        }
                    }
                }

                // Use accumulated F from this keyframe interval
                const F_for_rts = F_acc.map((r) => [...r]);

                steps.push({
                    x: { p: [...eskf.p], v: [...eskf.v], q: [...eskf.q], ba: [...eskf.ba], bg: [...eskf.bg], tUs: nowUs,
                        ...(useMag ? { mEarth: eskf.mEarth ? [...eskf.mEarth] : undefined, mBody: eskf.mBody ? [...eskf.mBody] : undefined } : {}),
                        ...(useTau ? { tauGps: eskf.tauGps } : {}),
                        ...(useKI ? { kI: eskf.kI ? [...eskf.kI] : undefined } : {}) },
                    P: eskf.P.map((r) => [...r]),
                    xPred: { p: [...xPred.p], v: [...xPred.v], q: [...xPred.q], ba: [...xPred.ba], bg: [...xPred.bg], tUs: xPred.tUs,
                        ...(useMag ? { mEarth: xPred.mEarth ? [...xPred.mEarth] : undefined, mBody: xPred.mBody ? [...xPred.mBody] : undefined } : {}),
                        ...(useTau ? { tauGps: xPred.tauGps } : {}),
                        ...(useKI ? { kI: xPred.kI ? [...xPred.kI] : undefined } : {}) },
                    PPred,
                    F: F_for_rts,
                    hasUpdate,
                });

                // Reset accumulated F for next interval
                F_acc = buildIdentityF(eskf.dim);

                nextKfUs += outputIntervalUs;
            }

            imuIdx++;
        }

        // ---- RTS backward smooth ----
        const filterResults = steps.map((s) => ({
            x: s.x,
            P: s.P,
            xPred: s.xPred,
            PPred: s.PPred,
        }));
        const Fmatrices = steps.slice(1).map((s) => s.F);
        const smoothed = rtsSmooth(filterResults, Fmatrices);

        // ---- Convert to output ----
        poses = smoothed.map((s) => ({
            tUs: s.x.tUs,
            p: [...s.x.p],
            v: [...s.x.v],
            q: [...s.x.q],
            ba: s.x.ba ? [...s.x.ba] : undefined,
            bg: s.x.bg ? [...s.x.bg] : undefined,
            P: s.P.map((r) => [...r]),
            mEarth: s.x.mEarth ? [...s.x.mEarth] : undefined,
            mBody: s.x.mBody ? [...s.x.mBody] : undefined,
            tauGps: s.x.tauGps,
            kI: s.x.kI ? [...s.x.kI] : undefined,
        }));

        // Re-seed for next iteration
        if (poses.length > 0 && iter < maxIter - 1) {
            const first = smoothed[0];
            p0 = first.x.p;
            v0 = first.x.v;
            q0 = first.x.q;
        }
    }

    return { smoothed: poses, t0Us, lat0, lon0, alt0 };
}

/**
 * Run the estimation pipeline and return a full PoseTrack IR.
 *
 * This is the canonical entry point. Returns the rich PoseTrack with
 * full state, covariance, meta, provenance, and the interpolating sampleAt().
 * The PoseTrack carries raw smoother NED p/v (no geodetic round-trip).
 *
 * See also: estimatePoses() for the flat legacy output (backward-compat wrapper).
 *
 * @param {object} data - same as estimatePoses
 * @param {object} origin - {lat, lon, alt}
 * @param {object} [opts] - same as estimatePoses
 * @returns {object} PoseTrack
 */
export function estimatePoseTrack(data, origin, opts = {}) {
    const { smoothed, lat0, lon0, alt0 } = _runEstimation(data, origin, opts);

    // Build PoseTrack samples directly from raw smoothed NED state
    const trackSamples = smoothed.map((s) => {
        const lla = nedToLlh({ n: s.p[0], e: s.p[1], d: s.p[2] }, lat0, lon0, alt0);

        // Extract 3×3 position covariance from full P (indices 0-2)
        const covPos = [
            s.P[0].slice(0, 3),
            s.P[1].slice(0, 3),
            s.P[2].slice(0, 3),
        ];

        // Extract 3×3 attitude covariance from full P (indices 6-8)
        const covAtt = [
            s.P[6].slice(6, 9),
            s.P[7].slice(6, 9),
            s.P[8].slice(6, 9),
        ];

        return {
            tUs: s.tUs,
            p: s.p,       // raw smoother NED — no geodetic round-trip
            v: s.v,
            q: s.q,
            lla,
            covPos,
            covAtt,
        };
    });

    // Converged nuisance-parameter estimates: average over the last quarter of
    // the trajectory (after the filter has had time to observe them). Enables
    // recovery validation (k_I / τ_gps) and downstream reuse.
    const estimatedParams = computeConvergedParams(smoothed);

    return createPoseTrack({
        samples: trackSamples,
        georefOrigin: { lat: origin.lat, lon: origin.lon, alt: origin.alt },
        source: {
            log: "Betaflight BBL",
            magModelSchema: opts.magModel ? String(opts.magModel.version || "2.x") : "none",
            solverConfig: {
                outputHz: opts.outputHz || 20,
                gpsPosSigma: opts.gpsPosSigma || 2.5,
                gpsVelSigma: opts.gpsVelSigma || 0.5,
                baroSigma: opts.baroSigma || 1.0,
                attSigma: opts.attSigma || 0.1,
                useMag: !!(opts.magModel && opts.magModel.earthFieldNedGauss),
            },
            estimatedParams,
        },
    });
}

/**
 * Average the estimated nuisance parameters over the last quarter of the
 * trajectory (post-convergence). Returns only the fields that were estimated.
 */
function computeConvergedParams(smoothed) {
    const n = smoothed.length;
    if (n === 0) return {};
    const start = Math.floor(n * 0.75);
    const tail = smoothed.slice(start);
    const out = {};
    if (tail[0].ba) {
        const acc = [0, 0, 0];
        for (const s of tail) for (let i = 0; i < 3; i++) acc[i] += s.ba[i];
        out.ba = acc.map((x) => x / tail.length);
    }
    if (tail[0].bg) {
        const acc = [0, 0, 0];
        for (const s of tail) for (let i = 0; i < 3; i++) acc[i] += s.bg[i];
        out.bg = acc.map((x) => x / tail.length);
    }
    if (tail[0].kI) {
        const acc = [0, 0, 0];
        for (const s of tail) for (let i = 0; i < 3; i++) acc[i] += s.kI[i];
        out.kI = acc.map((x) => x / tail.length);
    }
    if (tail[0].tauGps != null) {
        let acc = 0;
        for (const s of tail) acc += s.tauGps;
        out.tauGps = acc / tail.length;
    }
    if (tail[0].mEarth) {
        const acc = [0, 0, 0];
        for (const s of tail) for (let i = 0; i < 3; i++) acc[i] += s.mEarth[i];
        out.mEarth = acc.map((x) => x / tail.length);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function buildIdentityF(n) {
    const F = new Array(n);
    for (let i = 0; i < n; i++) {
        F[i] = new Array(n).fill(0);
        F[i][i] = 1;
    }
    return F;
}

function matMulFn(A, B) {
    const n = A.length;
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Array(n).fill(0);
        for (let k = 0; k < n; k++) {
            const aik = A[i][k];
            if (aik === 0) continue;
            for (let j = 0; j < n; j++) C[i][j] += aik * B[k][j];
        }
    }
    return C;
}

function findBaroAtTime(baro, tUs) {
    if (baro.length === 0) return null;
    let best = baro[0];
    let bestDt = Math.abs(baro[0].tUs - tUs);
    for (let i = 1; i < baro.length; i++) {
        const dt = Math.abs(baro[i].tUs - tUs);
        if (dt < bestDt) { bestDt = dt; best = baro[i]; }
    }
    return best.alt;
}
