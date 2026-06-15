/**
 * CSV serializer — thin adapter over the PoseTrack IR.
 *
 * Flat row per sample: tMs, lat, lon, altMsl, qw, qx, qy, qz,
 * vn, ve, vd, sigmaPos, sigmaAtt.
 */

/**
 * Serialize a PoseTrack to CSV string.
 *
 * @param {object} poseTrack - PoseTrack IR
 * @param {object} [opts]
 * @param {string} [opts.delimiter=","] - column delimiter
 * @param {boolean} [opts.header=true] - include header row
 * @param {number} [opts.precision=6] - decimal places
 * @returns {string} CSV string
 */
export function poseTrackToCsv(poseTrack, opts = {}) {
    const {
        delimiter = ",",
        header = true,
        precision = 6,
    } = opts;

    const { samples } = poseTrack;
    const lines = [];

    // Header
    if (header) {
        const cols = [];
        cols.push("tMs");
        cols.push("lat");
        cols.push("lon");
        cols.push("altMsl");
        cols.push("qw");
        cols.push("qx");
        cols.push("qy");
        cols.push("qz");
        cols.push("vn");
        cols.push("ve");
        cols.push("vd");
        cols.push("sigmaPos");
        cols.push("sigmaAtt");
        lines.push(cols.join(delimiter));
    }

    const t0Us = samples.length > 0 ? samples[0].tUs : 0;

    for (const s of samples) {
        const row = [];
        row.push(((s.tUs - t0Us) / 1000).toFixed(precision));
        row.push(s.lla ? s.lla.lat.toFixed(precision) : "");
        row.push(s.lla ? s.lla.lon.toFixed(precision) : "");
        row.push(s.lla ? s.lla.alt.toFixed(precision) : "");
        row.push(s.q[0].toFixed(precision));
        row.push(s.q[1].toFixed(precision));
        row.push(s.q[2].toFixed(precision));
        row.push(s.q[3].toFixed(precision));
        row.push(s.v[0].toFixed(precision));
        row.push(s.v[1].toFixed(precision));
        row.push(s.v[2].toFixed(precision));

        // sigmaPos from covPos trace
        let sigmaPos = 0;
        if (s.covPos) {
            sigmaPos = Math.sqrt(Math.max(0, (s.covPos[0][0] + s.covPos[1][1] + s.covPos[2][2]) / 3));
        }
        row.push(sigmaPos.toFixed(precision));

        // sigmaAtt from covAtt trace
        let sigmaAtt = 0;
        if (s.covAtt) {
            sigmaAtt = Math.sqrt(Math.max(0, (s.covAtt[0][0] + s.covAtt[1][1] + s.covAtt[2][2]) / 3)) * (180 / Math.PI);
        }
        row.push(sigmaAtt.toFixed(precision));

        lines.push(row.join(delimiter));
    }

    return `${lines.join("\n")  }\n`;
}
