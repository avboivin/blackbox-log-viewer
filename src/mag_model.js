/**
 * Magnetometer characterization model loader.
 * Parses the v2 JSON schema produced by the Betaflight Configurator mag characterization wizard.
 *
 * Schema: https://betaflight.com/blackbox/mag-characterization-model/2.0
 *
 * @module mag_model
 */

import { eulerToMatrix, mat3transpose, mat3mulVec, undoRollPitch, ALIGNMENT_MATRICES } from "./mag_alignment.js";

/**
 * Load and validate a characterization model from parsed JSON.
 *
 * @param {object} json - Parsed characterization_model JSON (v2 schema)
 * @returns {{ valid: boolean, error?: string, model?: MagModel }}
 */
export function loadMagCharacterizationModel(json) {
    if (!json || json.version !== "2.0") {
        return { valid: false, error: "Invalid model version. Expected v2.0." };
    }

    const ec = json.ellipsoid_correction;
    const align = json.alignment;
    const geo = json.geo_reference;

    if (!ec) {
        return { valid: false, error: "Model missing ellipsoid_correction." };
    }
    if (!align) {
        return { valid: false, error: "Model missing alignment data." };
    }
    if (!geo || geo.field_strength_nt == null) {
        return { valid: false, error: "Model missing geo_reference with field strength." };
    }

    // Build alignment matrix from model
    let alignmentMatrix;
    if (align.preset === 9 && align.euler_zyx_deg) {
        const e = align.euler_zyx_deg;
        alignmentMatrix = eulerToMatrix(e.roll, e.pitch, e.yaw);
    } else if (align.preset >= 1 && align.preset <= 8 && ALIGNMENT_MATRICES[align.preset]) {
        alignmentMatrix = ALIGNMENT_MATRICES[align.preset];
    } else {
        alignmentMatrix = ALIGNMENT_MATRICES[1]; // CW0 identity fallback
    }

    // Build world magnetic field unit vector from geo reference (NED frame)
    const DECL = Math.PI / 180;
    const incRad = geo.inclination_deg * DECL;
    const decRad = geo.declination_deg * DECL;
    const B_total = geo.field_strength_nt;
    const B_h = B_total * Math.cos(incRad);
    const B_world_ned = [
        B_h * Math.cos(decRad),
        B_h * Math.sin(decRad),
        B_total * Math.sin(incRad),
    ];
    const B_unit_ned = [
        B_world_ned[0] / B_total,
        B_world_ned[1] / B_total,
        B_world_ned[2] / B_total,
    ];

    const model = {
        version: "2.0",
        ellipsoid: {
            center: { x: ec.center.x, y: ec.center.y, z: ec.center.z },
            W_inv: ec.soft_iron,
            radius: ec.radius,
            residual_rms: ec.residual_rms,
        },
        alignment: {
            preset: align.preset,
            matrix: alignmentMatrix,
            euler: align.euler_zyx_deg || null,
        },
        geoReference: {
            declination: geo.declination_deg,
            inclination: geo.inclination_deg,
            fieldStrength: geo.field_strength_nt,
            B_unit_ned,
        },
        quality: json.quality
            ? {
                score: json.quality.score_percent,
                residualZ: json.quality.residual_z_rms,
                residualXY: json.quality.residual_xy_rms,
                fieldConsistency: json.quality.field_consistency_pct,
                chirality: json.quality.chirality_flag,
            }
            : null,
        poses: json.poses
            ? json.poses.map((p) => ({
                orientation: p.body_orientation,
                direction: p.cardinal_direction,
                qualityWeight: p.heading_quality_weight,
            }))
            : [],
    };

    return { valid: true, model };
}

/**
 * @typedef {object} MagModel
 * @property {string} version
 * @property {{ center: {x:number,y:number,z:number}, W_inv: number[3][3], radius: number, residual_rms: number }} ellipsoid
 * @property {{ preset: number, matrix: number[3][3], euler: {roll:number,pitch:number,yaw:number}|null }} alignment
 * @property {{ declination: number, inclination: number, fieldStrength: number, B_unit_ned: number[3] }} geoReference
 * @property {{ score: number, residualZ: number, residualXY: number, fieldConsistency: number, chirality: boolean }|null} quality
 * @property {Array<{ orientation: string, direction: string, qualityWeight: number }>} poses
 */
