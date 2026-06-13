/**
 * PoseSample — the neutral per-instant output record.
 *
 * The estimator emits an array of these. Each serializer (KML, CSV, GPX) is a thin
 * function over this array — no format abstraction layer.
 *
 * The body-to-world attitude quaternion follows the Betaflight convention:
 * body(FRD) → world(NED), Hamilton, scalar-first [w, x, y, z].
 *
 * @typedef {object} PoseSample
 * @property {number} tMs       - time since log start, milliseconds
 * @property {number} lat       - WGS84 latitude, degrees
 * @property {number} lon       - WGS84 longitude, degrees
 * @property {number} altMsl    - altitude MSL, metres
 * @property {[number,number,number,number]} q - attitude quaternion [w,x,y,z]
 * @property {[number,number,number]} vNed     - velocity NED, m/s
 * @property {number} sigmaPos  - 1-sigma position uncertainty, metres
 * @property {number} sigmaAtt  - 1-sigma attitude uncertainty, degrees
 */

/**
 * Create a PoseSample with default values.
 * @returns {PoseSample}
 */
export function createPoseSample() {
    return {
        tMs: 0,
        lat: 0,
        lon: 0,
        altMsl: 0,
        q: [1, 0, 0, 0],
        vNed: [0, 0, 0],
        sigmaPos: 0,
        sigmaAtt: 0,
    };
}
