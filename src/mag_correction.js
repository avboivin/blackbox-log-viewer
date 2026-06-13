/**
 * Per-sample magnetometer correction using a characterization model.
 *
 * Pipeline:
 *   1. m_c = W_inv × (m_raw − center)      — ellipsoid correction to unit sphere
 *   2. m_b = alignment × m_c                — sensor-to-body rotation
 *   3. m_leveled = undoRollPitch(m_b, r, p) — level to horizontal plane
 *   4. h_mag = atan2(−m_leveled[1], m_leveled[0]) — magnetic heading
 *   5. weight = cos(inclination) × sin(dip) — heading quality weight
 */


import { mat3mulVec, undoRollPitch } from "./mag_alignment.js";

/**
 * Apply ellipsoid correction to a raw mag reading.
 * m_clean = W_inv × (m_raw − center)
 */
function applyEllipsoidCorrection(raw, { center, W_inv }) {
    const dx = raw[0] - center.x;
    const dy = raw[1] - center.y;
    const dz = raw[2] - center.z;
    return [
        W_inv[0][0] * dx + W_inv[0][1] * dy + W_inv[0][2] * dz,
        W_inv[1][0] * dx + W_inv[1][1] * dy + W_inv[1][2] * dz,
        W_inv[2][0] * dx + W_inv[2][1] * dy + W_inv[2][2] * dz,
    ];
}

/**
 * Compute the heading quality weight based on local field geometry.
 * weight = cos(inclination) × sin(dip)
 *
 * The weight reflects how well the horizontal field component can be resolved
 * at this location — higher inclination means a weaker horizontal component.
 *
 * @param {MagModel} model - Characterization model with geoReference
 * @param {number[3]} m_body - Body-frame mag after alignment correction
 * @returns {number} weight in [0, 1]
 */
function computeHeadingWeight(model, m_body) {
    const { inclination, B_unit_ned } = model.geoReference;
    const cosI = Math.cos((inclination * Math.PI) / 180);
    if (cosI <= 0) return 0;

    const magNorm = Math.hypot(m_body[0], m_body[1], m_body[2]);
    if (magNorm < 1e-6) return 0;

    const B_body_unit = B_unit_ned;

    const dipBody = Math.acos(Math.abs(B_body_unit[2]));
    const sinDip = Math.sin(dipBody);

    return cosI * sinDip;
}

/**
 * Apply the full mag correction pipeline for a single sample.
 *
 * @param {number[3]} magRaw - Raw mag ADC values from the log [x, y, z]
 * @param {number} rollRad - Roll angle in radians
 * @param {number} pitchRad - Pitch angle in radians
 * @param {MagModel} model - Loaded characterization model
 * @returns {{ heading: number, weight: number, magCorrected: number[3] }|null}
 */
export function correctMagSample(magRaw, rollRad, pitchRad, model) {
    if (!magRaw || magRaw.length < 3) return null;
    if (magRaw[0] === 0 && magRaw[1] === 0 && magRaw[2] === 0) return null;

    // Step 1: Ellipsoid correction
    const mCorrected = applyEllipsoidCorrection(magRaw, model.ellipsoid);

    // Step 2: Sensor → body alignment
    const mBody = mat3mulVec(model.alignment.matrix, mCorrected);

    // Step 3: Level to horizontal frame
    const mLeveled = undoRollPitch(mBody, rollRad, pitchRad);

    // Step 4: NED heading (atan2 of -East / North)
    const hMag = Math.atan2(-mLeveled[1], mLeveled[0]);

    // Step 5: Analytic fusion weight
    const weight = computeHeadingWeight(model, mBody);

    return {
        heading: hMag,
        weight,
        magCorrected: mCorrected,
    };
}

/**
 * Compute the fused heading using phasor fusion of mag + GPS headings.
 * Avoids the 0°/360° singularity via atan2 blending.
 *
 * @param {number} hMag - Magnetic heading in radians
 * @param {number} weight - Mag weight [0,1]
 * @param {number} hGps - GPS heading in radians (or null if unavailable)
 * @returns {number} Fused heading in radians
 */
export function fuseHeading(hMag, weight, hGps) {
    if (hGps == null || weight >= 1.0) return hMag;
    if (weight <= 0) return hGps;

    const x = weight * Math.cos(hMag) + (1 - weight) * Math.cos(hGps);
    const y = weight * Math.sin(hMag) + (1 - weight) * Math.sin(hGps);
    return Math.atan2(y, x);
}
