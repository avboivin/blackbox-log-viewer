/**
 * Lossless JSON serializer for the PoseTrack IR.
 *
 * This is the canonical on-disk form — frame-explicit, self-describing,
 * carrying full state + covariance + provenance. Any external tool or
 * future adapter can consume the full result without re-running the estimator.
 *
 * Round-trip test: deserialize → serialize → compare byte-for-byte.
 */

import { createPoseTrack } from "../poseTrack.js";

/**
 * Serialize a PoseTrack to a JSON string.
 *
 * @param {object} poseTrack - PoseTrack IR
 * @param {number} [indent=2] - JSON indentation
 * @returns {string} JSON string
 */
export function poseTrackToJson(poseTrack, indent = 2) {
    return JSON.stringify(poseTrack, null, indent);
}

/**
 * Deserialize a PoseTrack from a JSON string.
 *
 * Rebuilds via createPoseTrack to get the full interpolating sampleAt
 * with slerp/lerp — identical to the in-memory accessor. There is no
 * circular import (this module is imported by consumers, not by poseTrack.js).
 *
 * @param {string} json - serialized PoseTrack
 * @returns {object} PoseTrack IR (with sampleAt bound)
 */
export function poseTrackFromJson(json) {
    const data = JSON.parse(json);

    if (!data.meta || !data.samples) {
        throw new Error("Invalid PoseTrack JSON: missing meta or samples");
    }

    return createPoseTrack({
        samples: data.samples,
        georefOrigin: data.meta.georefOrigin,
        source: data.meta.source,
    });
}
