/**
 * Flight ingestion — parses a Betaflight BBL log into the estimator data contract.
 *
 * Pure, framework-free module. Takes a FlightLog instance (which wraps the BFL
 * parser) and extracts IMU, GPS, baro, quaternion, and magnetometer streams
 * with all unit conversions applied per the ingestion field reference (planv5/03).
 *
 * The output is the canonical data contract consumed by estimatePoses() and the
 * Phase-0 sanity checks. No Vue/DOM/browser dependencies.
 */

import { FlightLogParser } from "../flightlog_parser.js";
import { quatToRot, eulerToQuat } from "./imuMechanization.js";
import { correctMagToBody } from "../mag_correction.js";

/**
 * Apply the mag characterization model to a raw ADC mag stream, producing
 * body-frame Gauss vectors for 3-axis ESKF fusion.
 *
 * Delegates correction math to mag_correction.js (planv5 Q2: single source of
 * truth). Does NO correction math itself — only maps over the stream.
 *
 * Samples that fail ellipsoid correction (zero vector, NaN) are dropped.
 *
 * @param {Array<{tUs:number, meas:[3]}>} magRaw - raw ADC mag samples from ingestFlightLog
 * @param {object} model - loaded MagModel (the .model from loadMagCharacterizationModel)
 * @returns {Array<{tUs:number, meas:[3]}>} body-frame Gauss vectors
 */
export function correctMagStream(magRaw, model) {
    const out = [];
    for (const m of magRaw) {
        const r = correctMagToBody(m.meas, model);
        if (!r) continue;
        const gpu = r.gaussPerCorrectedUnit;
        if (gpu != null && Math.abs(gpu) > 1e-12) {
            // Scale unit-sphere vector to physical Gauss
            out.push({
                tUs: m.tUs,
                meas: [r.mBody[0] * gpu, r.mBody[1] * gpu, r.mBody[2] * gpu],
            });
        }
    }
    return out;
}

const FLIGHT_LOG_FIELD_INDEX_TIME = FlightLogParser.prototype.FLIGHT_LOG_FIELD_INDEX_TIME;

/**
 * Parse a BFL log into the estimator input contract.
 *
 * @param {object} flightLog - FlightLog instance (from flightlog.js FlightLog constructor)
 * @param {object} [opts]
 * @param {number} [opts.startUs=0] - start time in microseconds
 * @param {number} [opts.endUs] - end time (default: all)
 * @param {number} [opts.gpsDecimation=1] - only take every Nth GPS fix
 * @returns {{
 *   imu: Array<{tUs:number, gyro:[3], accel:[3]}>,
 *   gps: Array<{tUs:number, lat:number, lon:number, alt:number, velNed:[3]|null, speed:number, course:number, numSat:number}>,
 *   baro: Array<{tUs:number, alt:number}>,
 *   quat: Array<{tUs:number, q:[4]}>,
 *   mag: Array<{tUs:number, meas:[3]}>,
 *   current: Array<{tUs:number, amps:number}>,
 *   gpsHome: {lat:number, lon:number, alt:number}|null,
 *   sysConfig: object,
 *   fieldPresence: {mag:boolean, quat:boolean, gpsVelned:boolean, baro:boolean, current:boolean},
 *   stats: {totalTimeUs:number, imuRateHz:number},
 * }}
 */
export function ingestFlightLog(flightLog, opts = {}) {
    const {
        startUs = 0,
        endUs = Number.MAX_SAFE_INTEGER,
        gpsDecimation = 1,
    } = opts;

    const sysConfig = flightLog.getSysConfig();
    const acc1G = sysConfig.acc_1G || 2048;
    // The parser normalizes sysConfig.gyroScale to radians-per-MICROSECOND per raw
    // count (flightlog_parser.js gyroScaleHandler: `*= (PI/180) * 1e-6`). The
    // estimator's IMU mechanization integrates angular rate in rad/SECOND, so we
    // must multiply by 1e6 (µs→s). Omitting this factor scales gyro to ~0 rad/s,
    // which silently freezes all reconstructed attitude while GPS-driven position
    // still looks fine — see 18 §29. (Cf. flightlog.js:1468, which recovers deg/s
    // as `gyroScale * 1e6 / (PI/180)`.)
    const gyroScale = (sysConfig.gyroScale || 1.0) * 1e6;

    // Resolve field indices
    const idx = (n) => flightLog.getMainFieldIndexByName(n);

    const idxTime = FLIGHT_LOG_FIELD_INDEX_TIME;
    const idxGyro = [idx("gyroADC[0]"), idx("gyroADC[1]"), idx("gyroADC[2]")];
    const idxAccel = [idx("accSmooth[0]"), idx("accSmooth[1]"), idx("accSmooth[2]")];
    const idxMag = [idx("magADC[0]"), idx("magADC[1]"), idx("magADC[2]")];
    const idxQuat = [idx("imuQuaternion[0]"), idx("imuQuaternion[1]"), idx("imuQuaternion[2]")];
    const idxBaro = idx("baroAlt");
    const idxGpsLat = idx("GPS_coord[0]");
    const idxGpsLon = idx("GPS_coord[1]");
    const idxGpsAlt = idx("GPS_altitude");
    const idxGpsSpeed = idx("GPS_speed");
    const idxGpsCourse = idx("GPS_ground_course");
    const idxGpsVelned = [idx("GPS_velned[0]"), idx("GPS_velned[1]"), idx("GPS_velned[2]")];
    const idxGpsNumSat = idx("GPS_numSat");
    // GPS home is an H-FRAME field, NOT a main I/P-frame field — getMainFieldIndexByName
    // cannot see it (it would return undefined and silently yield gpsHome=null). The
    // viewer decodes the H-frame into the intraframe directory; read it via getGPSHome().
    // See planv5/03 (frame-stream reference) and flightlog.js getGPSHome().
    const gpsHome = typeof flightLog.getGPSHome === "function" ? flightLog.getGPSHome() : null;
    const idxCurrent = idx("amperageLatest");

    const hasGyro = idxGyro.every((i) => i != null);
    const hasAccel = idxAccel.every((i) => i != null);
    const hasMag = idxMag.every((i) => i != null);
    const hasQuat = idxQuat.every((i) => i != null);
    const hasBaro = idxBaro != null;
    const hasGpsPos = idxGpsLat != null && idxGpsLon != null;
    const hasGpsVelned = idxGpsVelned.every((i) => i != null);
    const hasCurrent = idxCurrent != null;

    const fieldPresence = {
        mag: hasMag,
        quat: hasQuat,
        gpsVelned: hasGpsVelned,
        baro: hasBaro,
        current: hasCurrent,
        gps: hasGpsPos,
        gyro: hasGyro,
        accel: hasAccel,
    };

    // Accumulators
    const imu = [];
    const gps = [];
    const baro = [];
    const quat = [];
    const magRaw = [];
    const currentSamples = [];
    let gpsCounter = 0;

    // Track GPS state between frames to avoid duplicates
    let lastGpsLat = null, lastGpsLon = null, lastGpsAlt = null;

    const chunks = flightLog.getChunksInTimeRange(startUs, endUs);

    for (const chunk of chunks) {
        for (const frame of chunk.frames) {
            const tUs = frame[idxTime];
            if (tUs < startUs || tUs > endUs) continue;

            // ---- IMU (gyro + accel) ----
            if (hasGyro && hasAccel) {
                const gyroRad = [
                    frame[idxGyro[0]] * gyroScale,
                    frame[idxGyro[1]] * gyroScale,
                    frame[idxGyro[2]] * gyroScale,
                ];
                const accelMs2 = [
                    (frame[idxAccel[0]] / acc1G) * 9.80665,
                    (frame[idxAccel[1]] / acc1G) * 9.80665,
                    (frame[idxAccel[2]] / acc1G) * 9.80665,
                ];
                imu.push({ tUs, gyro: gyroRad, accel: accelMs2 });
            }

            // ---- Barometer ----
            if (hasBaro) {
                const baroRaw = frame[idxBaro];
                if (baroRaw != null) {
                    baro.push({ tUs, alt: baroRaw / 100 });
                }
            }

            // ---- FC Quaternion ----
            // Betaflight logs the attitude quaternion body(FRD)→world(NED),
            // Hamilton scalar-first [w,x,y,z]. The quaternion components are stored
            // as int16 scaled by 1/32767; qw is reconstructed from the unit-norm
            // constraint qw = √(1 − qx² − qy² − qz²) with qw ≥ 0.
            //
            // CONVENTION FIX (2026-06-15): Betaflight applies the quaternion as
            //   v_world = q_bf* ⊗ v_body ⊗ q_bf
            // while our ESKF uses the standard Hamilton product:
            //   v_world = q_our ⊗ v_body ⊗ q_our*
            // The same numeric quaternion produces opposite rotations in these two
            // conventions. The fix: conjugate the BF quaternion to convert between
            // conventions (q_our = q_bf*), then extract Euler using standard formulas.
            // The BF yaw formula (−atan2) compensates for the conjugate's yaw flip
            // while producing correct pitch/roll signs for the FRD/NED frame.
            if (hasQuat) {
                const qx = frame[idxQuat[0]] / 32767;
                const qy = frame[idxQuat[1]] / 32767;
                const qz = frame[idxQuat[2]] / 32767;
                const m = qx * qx + qy * qy + qz * qz;
                let qw;
                if (m < 1.0) {
                    qw = Math.sqrt(1.0 - m);
                } else {
                    qw = 0;
                }
                // Partial conjugate: negate qy (pitch) and qz (yaw) from raw quaternion.
                // qx (roll) must NOT be negated — the full conjugate [qw,-qx,-qy,-qz]
                // inverts east/west permanently (dead end: §42.13). The un-negated qx
                // creates a roll-coupling artifact during banked turns, cured by
                // skipping the quat-prior entirely during high-rate maneuvers
                // (estimatorLoop.js sharp-turn gate). Pitch and yaw are correct in
                // both static and dynamic conditions with this combination.
                const qFixed = [qw, qx, -qy, -qz];
                quat.push({ tUs, q: qFixed });
            }

            // ---- Magnetometer (raw ADC, later converted through model) ----
            if (hasMag) {
                magRaw.push({
                    tUs,
                    meas: [frame[idxMag[0]], frame[idxMag[1]], frame[idxMag[2]]],
                });
            }

            // ---- Current ----
            if (hasCurrent) {
                const rawCurrent = frame[idxCurrent];
                if (rawCurrent != null) {
                    // amperageLatest is in centiamps, convert to amps
                    currentSamples.push({ tUs, amps: rawCurrent / 100 });
                }
            }

            // ---- GPS detection ----
            if (hasGpsPos) {
                const lat = frame[idxGpsLat] / 1e7;
                const lon = frame[idxGpsLon] / 1e7;
                const altMsl = frame[idxGpsAlt] != null ? frame[idxGpsAlt] / 10 : null;

                // Valid GPS: non-zero lat/lon
                if (lat !== 0 && lon !== 0) {
                    const changed = lat !== lastGpsLat || lon !== lastGpsLon || altMsl !== lastGpsAlt;
                    if (changed) {
                        lastGpsLat = lat;
                    lastGpsLon = lon;
                    lastGpsAlt = altMsl;

                        gpsCounter++;
                        if (gpsCounter % gpsDecimation !== 0) continue;

                        let velNed = null;
                        if (hasGpsVelned) {
                            velNed = [
                                frame[idxGpsVelned[0]] / 100,
                                frame[idxGpsVelned[1]] / 100,
                                frame[idxGpsVelned[2]] / 100,
                            ];
                        }

                        const speed = frame[idxGpsSpeed] != null ? frame[idxGpsSpeed] / 100 : 0;
                        const course = frame[idxGpsCourse] != null ? frame[idxGpsCourse] / 10 : 0;
                        const numSat = frame[idxGpsNumSat] != null ? frame[idxGpsNumSat] : 0;

                        // Reject corrupted numSat
                        const ns = numSat > 100 ? 0 : numSat;

                        gps.push({
                            tUs,
                            lat,
                            lon,
                            alt: altMsl,
                            velNed,
                            speed,
                            course,
                            numSat: ns,
                        });
                    }
                }
            }
        }
    }

    const totalTimeUs = imu.length > 0 ? imu[imu.length - 1].tUs - imu[0].tUs : 0;
    const imuRateHz = totalTimeUs > 0 ? (imu.length / (totalTimeUs / 1e6)) : 0;

    return {
        imu,
        gps,
        baro,
        quat,
        mag: magRaw,
        current: currentSamples,
        gpsHome,
        sysConfig,
        fieldPresence,
        stats: { totalTimeUs, imuRateHz },
    };
}

/**
 * Load a FlightLog from a Uint8Array buffer (for Node/headless usage).
 *
 * The FlightLog/ArrayDataStream expects indexable data via [] accessor,
 * so pass a Uint8Array (or Node Buffer which extends Uint8Array), not a bare ArrayBuffer.
 *
 * @param {Uint8Array|Buffer} logData - raw BFL file bytes
 * @returns {object} FlightLog instance
 */
export async function loadFlightLogFromBuffer(logData) {
    const { FlightLog } = await import("../flightlog.js");
    const fl = new FlightLog(logData);
    if (fl.openLog) fl.openLog(0);
    return fl;
}

/**
 * Load a FlightLog from a file path (Node only).
 *
 * @param {string} filePath - path to .BFL file
 * @returns {object} FlightLog instance
 */
export async function loadFlightLogFromFile(filePath) {
    const fs = await import("fs");
    const buffer = fs.readFileSync(filePath);
    return loadFlightLogFromBuffer(new Uint8Array(buffer));
}
