/**
 * Estimator loop — orchestrates forward ESKF pass, RTS backward smoothing,
 * and Gauss-Newton iteration to produce maximum-a-posteriori pose estimates.
 *
 * Schedules a keyframe at configurable rate. Between keyframes the forward
 * ESKF integrates IMU per-step. At each keyframe GPS/baro/quaternion updates
 * are applied. The backward RTS smoother then distributes corrections.
 */

import { createEskf, eskfPredict, eskfUpdate } from "./eskf.js";
import { createGpsPositionFactor, createGpsVelocityFactor, createBaroFactor, createQuaternionPrior, createMagFactor, createDeclinationFactor } from "./measurements.js";
import { rtsSmooth } from "./rtsSmoother.js";
import { llhToNed, nedToLlh } from "./geodesy.js";

/**
 * Run the estimation pipeline over pre-parsed sensor data.
 *
 * @param {object} data
 * @param {Array<{tUs:number, gyro:[3], accel:[3]}>} data.imu - sorted IMU samples
 * @param {Array<{tUs:number, lat:number, lon:number, alt:number, velNed:[3]?}>} data.gps - sorted GPS fixes
 * @param {Array<{tUs:number, alt:number}>} data.baro - sorted baro samples
 * @param {Array<{tUs:number, q:[4]}>} data.quat - sorted FC quaternion samples
 * @param {Array<{tUs:number, meas:[3]}>} [data.mag] - sorted 3-axis mag samples (body FRD, Gauss)
 * @param {object} origin - {lat, lon, alt} NED origin (first GPS home)
 * @param {object} [opts]
 * @param {number} [opts.outputHz=20] - keyframe output rate
 * @param {number} [opts.gpsPosSigma=2.5]
 * @param {number} [opts.gpsVelSigma=0.5]
 * @param {number} [opts.baroSigma=1.0]
 * @param {number} [opts.attSigma=0.1]
 * @param {number} [opts.magSigma=0.05] - mag measurement noise 1σ (Gauss)
 * @param {number} [opts.declSigma=0.34] - declination constraint 1σ (rad)
 * @param {object} [opts.magModel] - fusion block from mag characterization model
 * @returns {Array<{tMs:number, lat:number, lon:number, altMsl:number, q:[4], vNed:[3], sigmaPos:number, sigmaAtt:number}>}
 */
export function estimatePoses(data, origin, opts = {}) {
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
    } = opts;

    const { imu, gps, baro, quat, mag } = data;
    if (!imu || imu.length === 0) return [];

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
        const eskfOpts = { p0, v0, q0, sigmaPos: 5, sigmaVel: 2, sigmaAtt: 0.2 };
        if (useMag) {
            const me = magModel.earthFieldNedGauss;
            eskfOpts.mEarth0 = [me.n, me.e, me.d];
            eskfOpts.mBody0 = [0, 0, 0];
        }
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
                const xPred = { p: [...eskf.p], v: [...eskf.v], q: [...eskf.q], tUs: nowUs };
                const PPred = eskf.P.map((r) => [...r]);

                let hasUpdate = false;

                // GPS update if fix is available near this time
                while (gpsIdx < gps.length && gps[gpsIdx].tUs <= nextKfUs + outputIntervalUs * 0.5) {
                    const gpsF = gps[gpsIdx];
                    const gNed = llhToNed(gpsF.lat, gpsF.lon, gpsF.alt, lat0, lon0, alt0);
                    const fP = createGpsPositionFactor({ n: gNed.n, e: gNed.e, d: gNed.d }, gpsPosSigma);
                    if (eskfUpdate(eskf, fP, { n: gNed.n, e: gNed.e, d: gNed.d })) hasUpdate = true;
                    if (gpsF.velNed) {
                        const fV = createGpsVelocityFactor(
                            { n: gpsF.velNed[0], e: gpsF.velNed[1], d: gpsF.velNed[2] },
                            gpsVelSigma,
                        );
                        if (eskfUpdate(eskf, fV, { n: gpsF.velNed[0], e: gpsF.velNed[1], d: gpsF.velNed[2] }))
                            hasUpdate = true;
                    }
                    gpsIdx++;
                }

                // Baro update
                while (baroIdx < baro.length && baro[baroIdx].tUs <= nextKfUs) {
                    const fB = createBaroFactor(baro[baroIdx].alt, baroOffset, baroSigma);
                    if (eskfUpdate(eskf, fB, baro[baroIdx].alt)) hasUpdate = true;
                    baroIdx++;
                }

                // Quaternion prior
                while (quatIdx < quat.length && quat[quatIdx].tUs <= nextKfUs) {
                    const fQ = createQuaternionPrior(quat[quatIdx].q, attSigma);
                    if (eskfUpdate(eskf, fQ, quat[quatIdx].q)) hasUpdate = true;
                    quatIdx++;
                }

                // 3-axis mag update (gate 3.0 per 09 §1)
                if (useMag) {
                    while (magIdx < mag.length && mag[magIdx].tUs <= nextKfUs) {
                        const fM = createMagFactor(mag[magIdx].meas, magMeasSigma);
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
                    x: { p: [...eskf.p], v: [...eskf.v], q: [...eskf.q], tUs: nowUs },
                    P: eskf.P.map((r) => [...r]),
                    xPred: { p: [...xPred.p], v: [...xPred.v], q: [...xPred.q], tUs: xPred.tUs },
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
        const Fmatrices = steps.slice(0, steps.length - 1).map((s) => s.F);
        const smoothed = rtsSmooth(filterResults, Fmatrices);

        // ---- Convert to output ----
        poses = smoothed.map((s) => {
            const llh = nedToLlh({ n: s.x.p[0], e: s.x.p[1], d: s.x.p[2] }, lat0, lon0, alt0);
            return {
                tMs: (s.x.tUs - t0Us) / 1000,
                lat: llh.lat,
                lon: llh.lon,
                altMsl: llh.alt,
                q: s.x.q,
                vNed: s.x.v,
                sigmaPos: Math.sqrt(Math.max(0, (s.P[0][0] + s.P[1][1] + s.P[2][2]) / 3)),
                sigmaAtt: Math.sqrt(Math.max(0, (s.P[6][6] + s.P[7][7] + s.P[8][8]) / 3)) * (180 / Math.PI),
            };
        });

        // Re-seed for next iteration
        if (poses.length > 0 && iter < maxIter - 1) {
            const first = smoothed[0];
            p0 = first.x.p;
            v0 = first.x.v;
            q0 = first.x.q;
        }
    }

    return poses;
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
