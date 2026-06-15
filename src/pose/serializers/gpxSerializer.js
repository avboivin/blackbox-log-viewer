/**
 * GPX serializer — thin adapter over the PoseTrack IR.
 *
 * GPX has no native per-point orientation, so this is intentionally lossy:
 * it carries position + synthetic time only. The triad KML is the primary
 * attitude output.
 */

function esc(s) {
    return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

/**
 * Serialize a PoseTrack to GPX 1.1 string.
 *
 * @param {object} poseTrack - PoseTrack IR
 * @param {object} [opts]
 * @param {string} [opts.trackName="Betaflight Track"] - GPX track name
 * @returns {string} GPX XML string
 */
export function poseTrackToGpx(poseTrack, opts = {}) {
    const { trackName = "Betaflight Track" } = opts;
    const { samples } = poseTrack;
    if (!samples || samples.length === 0) {
        return '<?xml version="1.0" encoding="UTF-8"?><gpx version="1.1" creator="Betaflight Pose Estimator"><trk><name>Empty</name><trkseg></trkseg></trk></gpx>';
    }

    const t0Us = samples[0].tUs;

    const lines = [];
    lines.push('<?xml version="1.0" encoding="UTF-8"?>');
    lines.push('<gpx version="1.1" creator="Betaflight Pose Estimator" xmlns="http://www.topografix.com/GPX/1/1">');
    lines.push(`  <trk>`);
    lines.push(`    <name>${esc(trackName)}</name>`);
    lines.push(`    <trkseg>`);

    for (const s of samples) {
        if (!s.lla) continue;
        if (s.tUs == null) continue;
        const tMs = (s.tUs - t0Us) / 1000;
        const d = new Date(tMs);
        if (isNaN(d.getTime())) continue;
        const iso = d.toISOString();

        lines.push(`      <trkpt lat="${s.lla.lat}" lon="${s.lla.lon}">`);
        lines.push(`        <ele>${s.lla.alt}</ele>`);
        lines.push(`        <time>${iso}</time>`);
        lines.push(`      </trkpt>`);
    }

    lines.push(`    </trkseg>`);
    lines.push(`  </trk>`);
    lines.push(`</gpx>`);

    return `${lines.join("\n")  }\n`;
}
