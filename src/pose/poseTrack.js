/**
 * PoseTrack — the immutable intermediate representation (IR) for body-pose results.
 *
 * This is the single source of truth for all downstream consumers. Every serializer
 * reads from this; none of them shapes it. The IR carries full state, covariance,
 * provenance, and an interpolating accessor.
 *
 * Rule: nothing format-specific leaks above the serializer layer.
 */

export const POSE_TRACK_SCHEMA = 1;

/**
 * Spherical linear interpolation between two quaternions.
 * q1, q2 are [w, x, y, z] Hamilton.
 */
function slerp(q1, q2, t) {
    let cosTheta = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
    if (cosTheta < 0) {
        // Take the short path
        q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
        cosTheta = -cosTheta;
    }
    if (cosTheta > 0.9995) {
        // Linear interpolation for small angles
        const result = [
            q1[0] + t * (q2[0] - q1[0]),
            q1[1] + t * (q2[1] - q1[1]),
            q1[2] + t * (q2[2] - q1[2]),
            q1[3] + t * (q2[3] - q1[3]),
        ];
        const n = Math.sqrt(result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2);
        if (n < 1e-14) return [1, 0, 0, 0];
        return [result[0] / n, result[1] / n, result[2] / n, result[3] / n];
    }
    const theta0 = Math.acos(cosTheta);
    const sinTheta0 = Math.sin(theta0);
    const s0 = Math.sin((1 - t) * theta0) / sinTheta0;
    const s1 = Math.sin(t * theta0) / sinTheta0;
    return [
        s0 * q1[0] + s1 * q2[0],
        s0 * q1[1] + s1 * q2[1],
        s0 * q1[2] + s1 * q2[2],
        s0 * q1[3] + s1 * q2[3],
    ];
}

/**
 * Binary search for the sample at or immediately before tUs.
 * Returns the index of the rightmost sample where sample.tUs <= tUs.
 */
function findIndexBefore(samples, tUs) {
    let lo = 0, hi = samples.length - 1;
    if (tUs <= samples[lo].tUs) return lo;
    if (tUs >= samples[hi].tUs) return hi;
    while (lo < hi - 1) {
        const mid = (lo + hi) >> 1;
        if (samples[mid].tUs <= tUs) lo = mid;
        else hi = mid;
    }
    return lo;
}

/**
 * Create a PoseTrack from estimator output.
 *
 * @param {object} opts
 * @param {Array<{tUs:number, p:[3], v:[3], q:[4], covPos:number[][], covAtt:number[][], lla?:{lat,lon,alt}, diagnostics?:object}>} opts.samples
 * @param {{lat:number, lon:number, alt:number}} opts.georefOrigin
 * @param {object} opts.source - provenance info { log, magModelSchema, solverConfig }
 * @returns {PoseTrack}
 */
export function createPoseTrack({ samples, georefOrigin, source }) {
    // Sort by time (should already be sorted)
    const sorted = [...samples].sort((a, b) => a.tUs - b.tUs);

    const track = {
        meta: {
            schemaVersion: POSE_TRACK_SCHEMA,
            frame: "body=FRD, world=NED",
            georefOrigin: { ...georefOrigin },
            units: {
                pos: "m",
                vel: "m/s",
                attitude: "quaternion[w,x,y,z]",
                time: "us",
                covPos: "m^2 (3x3 NED)",
                covAtt: "rad^2 (3x3)",
            },
            source: { ...source },
        },

        samples: sorted,

        /**
         * Interpolating accessor — returns a PoseSample at arbitrary time tUs.
         *
         * Position/velocity: linear interpolation (lerp)
         * Attitude: spherical linear interpolation (slerp)
         * Covariance: linear interpolation between bounding samples
         * LLA: linear interpolation
         *
         * For tUs before the first sample or after the last sample, the
         * nearest sample is returned (no extrapolation).
         *
         * @param {number} tUs - query time in microseconds
         * @returns {object} PoseSample-like object
         */
        sampleAt(tUs) {
            if (sorted.length === 0) return null;
            if (sorted.length === 1) return sorted[0];

            if (tUs <= sorted[0].tUs) return sorted[0];
            if (tUs >= sorted[sorted.length - 1].tUs) return sorted[sorted.length - 1];

            const i = findIndexBefore(sorted, tUs);
            const j = i + 1;
            if (j >= sorted.length) return sorted[i];

            const si = sorted[i];
            const sj = sorted[j];
            const dt = sj.tUs - si.tUs;
            if (dt <= 0) return si;

            const t = (tUs - si.tUs) / dt;

            // Lerp position
            const p = [
                si.p[0] + t * (sj.p[0] - si.p[0]),
                si.p[1] + t * (sj.p[1] - si.p[1]),
                si.p[2] + t * (sj.p[2] - si.p[2]),
            ];

            // Lerp velocity
            const v = [
                si.v[0] + t * (sj.v[0] - si.v[0]),
                si.v[1] + t * (sj.v[1] - si.v[1]),
                si.v[2] + t * (sj.v[2] - si.v[2]),
            ];

            // Slerp quaternion
            const q = slerp(si.q, sj.q, t);

            // Lerp LLA if present
            let lla = null;
            if (si.lla && sj.lla) {
                lla = {
                    lat: si.lla.lat + t * (sj.lla.lat - si.lla.lat),
                    lon: si.lla.lon + t * (sj.lla.lon - si.lla.lon),
                    alt: si.lla.alt + t * (sj.lla.alt - si.lla.alt),
                };
            }

            // Lerp covariance
            const lerpCov = (ca, cb) => {
                if (!ca || !cb) return ca || cb;
                const n = ca.length;
                const result = new Array(n);
                for (let ri = 0; ri < n; ri++) {
                    result[ri] = new Array(n);
                    for (let ci = 0; ci < n; ci++) {
                        result[ri][ci] = ca[ri][ci] + t * (cb[ri][ci] - ca[ri][ci]);
                    }
                }
                return result;
            };
            const covPos = lerpCov(si.covPos, sj.covPos);
            const covAtt = lerpCov(si.covAtt, sj.covAtt);

            const result = {
                tUs,
                p,
                v,
                q,
                lla,
                covPos,
                covAtt,
            };

            // Carry diagnostics from the nearest sample
            if (si.diagnostics && t < 0.5) result.diagnostics = si.diagnostics;
            else if (sj.diagnostics) result.diagnostics = sj.diagnostics;

            return result;
        },
    };

    return track;
}

/**
 * PoseTrack type definition (for documentation).
 *
 * @typedef {object} PoseTrack
 * @property {object} meta
 * @property {number} meta.schemaVersion
 * @property {string} meta.frame
 * @property {{lat:number, lon:number, alt:number}} meta.georefOrigin
 * @property {object} meta.units
 * @property {object} meta.source
 * @property {PoseSampleInternal[]} samples - time-sorted
 * @property {function(number):PoseSampleInternal} sampleAt
 */

/**
 * Internal PoseSample used within the track.
 *
 * @typedef {object} PoseSampleInternal
 * @property {number} tUs - time in microseconds
 * @property {number[]} p - position NED [n,e,d] (m)
 * @property {number[]} v - velocity NED [vn,ve,vd] (m/s)
 * @property {number[]} q - attitude body→world [w,x,y,z]
 * @property {{lat:number,lon:number,alt:number}|null} lla - geodetic coords
 * @property {number[][]} covPos - 3×3 position covariance (m²)
 * @property {number[][]} covAtt - 3×3 attitude covariance (rad²)
 * @property {object} [diagnostics] - optional per-sample diagnostics
 */
