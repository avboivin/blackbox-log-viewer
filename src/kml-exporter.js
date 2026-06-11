/**
 * KML exporter with magnetometer-corrected heading.
 *
 * Produces KML 2.2 XML with Placemark LineString for the flight track,
 * extended data for each point including corrected heading.
 *
 * If a mag characterization model v2 is provided, the corrected heading is
 * computed per-sample using ellipsoid correction + alignment + leveling.
 *
 * @module kml-exporter
 */

import { correctMagSample } from "./mag_correction.js";

function escapeXml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function getFieldIdx(flightLog, name) {
    return flightLog.getMainFieldIndexByName(name);
}

function getFieldValue(frame, flightLog, name) {
    const idx = getFieldIdx(flightLog, name);
    if (idx == null || frame[idx] == null) return null;
    return frame[idx];
}

function getMagRaw(frame, flightLog) {
    const x = getFieldValue(frame, flightLog, "magADC[0]");
    const y = getFieldValue(frame, flightLog, "magADC[1]");
    const z = getFieldValue(frame, flightLog, "magADC[2]");
    if (x == null || y == null || z == null) return null;
    return [x, y, z];
}

export function exportKml(flightLog, magModel = null) {
    const gpsCoordIdx = getFieldIdx(flightLog, "GPS_coord[0]");
    const gpsAltIdx = getFieldIdx(flightLog, "GPS_altitude");
    if (gpsCoordIdx == null) return null;

    const lines = [];
    lines.push('<?xml version="1.0" encoding="UTF-8"?>');
    lines.push('<kml xmlns="http://www.opengis.net/kml/2.2">');
    lines.push("  <Document>");
    lines.push("    <name>Betaflight Track</name>");
    lines.push('    <Style id="trackStyle">');
    lines.push("      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>");
    lines.push("    </Style>");
    lines.push("    <Placemark>");
    lines.push("      <name>Flight Path</name>");
    lines.push("      <styleUrl>#trackStyle</styleUrl>");
    lines.push("      <LineString>");
    lines.push("        <extrude>0</extrude>");
    lines.push("        <altitudeMode>clampToGround</altitudeMode>");
    lines.push("        <coordinates>");

    const coords = [];
    const headingSamples = [];

    const chunks = flightLog.getChunksInTimeRange(0, Number.MAX_SAFE_INTEGER);
    for (const chunk of chunks) {
        for (const frame of chunk.frames) {
            if (frame[gpsCoordIdx] == null) continue;
            const lon = frame[gpsCoordIdx] / 1e7;
            const lat = frame[gpsCoordIdx + 1] / 1e7;
            const alt = gpsAltIdx != null && frame[gpsAltIdx] != null ? frame[gpsAltIdx] / 100 : 0;
            coords.push(`${lon.toFixed(7)},${lat.toFixed(7)},${alt.toFixed(1)}`);

            if (magModel) {
                const magRaw = getMagRaw(frame, flightLog);
                const roll = getFieldValue(frame, flightLog, "heading[0]");
                const pitch = getFieldValue(frame, flightLog, "heading[1]");
                if (magRaw && roll != null && pitch != null) {
                    const result = correctMagSample(magRaw, roll, pitch, magModel);
                    if (result) {
                        headingSamples.push({
                            coordIdx: coords.length - 1,
                            headingDeg: ((result.heading * 180) / Math.PI + 360) % 360,
                            weight: result.weight,
                        });
                    }
                }
            }
        }
    }

    if (coords.length === 0) return null;

    for (const c of coords) {
        lines.push(`          ${c}`);
    }
    lines.push("        </coordinates>");
    lines.push("      </LineString>");
    lines.push("    </Placemark>");

    if (magModel && headingSamples.length > 0) {
        lines.push("    <!-- Corrected heading per GPS sample -->");
        for (const hs of headingSamples) {
            const coord = coords[hs.coordIdx];
            lines.push("    <Placemark>");
            lines.push(`      <Point><coordinates>${coord}</coordinates></Point>`);
            lines.push("      <ExtendedData>");
            lines.push(`        <Data name="corrected_heading_deg"><value>${hs.headingDeg.toFixed(1)}</value></Data>`);
            lines.push(`        <Data name="heading_weight"><value>${hs.weight.toFixed(4)}</value></Data>`);
            lines.push("      </ExtendedData>");
            lines.push("    </Placemark>");
        }
    }

    lines.push("  </Document>");
    lines.push("</kml>");

    return lines.join("\n");
}

export class KmlExporter {
    constructor(flightLog, magModel = null) {
        this._flightLog = flightLog;
        this._magModel = magModel;
    }

    dump(callback) {
        try {
            const kml = exportKml(this._flightLog, this._magModel);
            callback(kml);
        } catch (e) {
            console.error("KML export failed", e);
            callback(null);
        }
    }
}
