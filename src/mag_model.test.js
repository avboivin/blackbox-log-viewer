import { describe, it, expect } from "vitest";
import { loadMagCharacterizationModel, computeMagQualityBounds } from "./mag_model.js";

describe("mag_model — schema 2.2 support", () => {
    it("loads a schema 2.2 model with existing downstream_fusion block", () => {
        const json = {
            version: "2.2",
            ellipsoid_correction: {
                center: { x: 100, y: 0, z: 0 },
                soft_iron: [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]],
                radius: 1.0,
                residual_rms: 0.01,
            },
            alignment: { preset: 1 },
            geo_reference: {
                declination_deg: 0,
                inclination_deg: 60,
                field_strength_nt: 50000,
            },
            quality: {
                score_percent: 95,
                residual_xy_rms: 0.02,
                residual_z_rms: 0.01,
                field_consistency_pct: 1.0,
                chirality_flag: false,
            },
            downstream_fusion: {
                frame: "FRD",
                nt_per_corrected_unit: 50000,
                gauss_per_corrected_unit: 0.5,
                earth_field_ned_gauss: { n: 0.25, e: 0, d: 0.433 },
                mag_noise_gauss: { sigma: 0.005, sigma_xy: 0.01, sigma_z: 0.005 },
                quality_bounds: {
                    field_strength_mg: 500,
                    field_strength_ok: true,
                    bounds_ok: true,
                },
            },
        };

        const result = loadMagCharacterizationModel(json);
        expect(result.valid).toBe(true);
        expect(result.model.version).toBe("2.2");
        expect(result.model.fusion).toBeDefined();
        expect(result.model.fusion.frame).toBe("FRD");
        expect(result.model.fusion.gaussPerCorrectedUnit).toBe(0.5);
        expect(result.model.fusion.earthFieldNedGauss).toEqual({ n: 0.25, e: 0, d: 0.433 });
        expect(result.model.fusion.magNoiseGauss.sigma).toBe(0.005);
    });

    it("computes downstream_fusion when absent (schema 2.1 model)", () => {
        const json = {
            version: "2.1",
            ellipsoid_correction: {
                center: { x: 700, y: 80, z: 120 },
                soft_iron: [
                    [0.00061, -0.000022, 0.000003],
                    [0, 0.00061, -0.00002],
                    [0, 0, 0.00062],
                ],
                radius: 1.0,
                residual_rms: 0.012,
            },
            alignment: { preset: 1 },
            geo_reference: {
                declination_deg: 0,
                inclination_deg: 70,
                field_strength_nt: 54000,
            },
            quality: {
                score_percent: 96,
                residual_xy_rms: 0.06,
                residual_z_rms: 0.016,
                field_consistency_pct: 4.7,
                chirality_flag: false,
            },
        };

        const result = loadMagCharacterizationModel(json);
        expect(result.valid).toBe(true);
        expect(result.model.version).toBe("2.1");
        expect(result.model.fusion).toBeDefined();
        expect(result.model.fusion.frame).toBe("FRD");

        // Computed from field_strength_nt / radius = 54000 / 1.0 = 54000
        // gauss_per_unit = 54000 / 1e5 = 0.54
        const gpu = result.model.fusion.gaussPerCorrectedUnit;
        expect(gpu).toBeCloseTo(0.54, 3);
        expect(gpu).toBeGreaterThan(0);

        // Earth field: bH = 0.54 * cos(70) ≈ 0.54 * 0.342 = 0.1847, n = bH * cos(0) = 0.1847
        expect(result.model.fusion.earthFieldNedGauss.n).toBeCloseTo(0.1847, 2);

        // Noise: sigma = residual * gauss_per_unit ≈ 0.012 * 0.54 = 0.00648
        expect(result.model.fusion.magNoiseGauss.sigma).toBeCloseTo(0.00648, 3);

        // Quality bounds
        expect(result.model.fusion.qualityBounds.bounds_ok).toBe(true);
        expect(result.model.fusion.qualityBounds.field_strength_ok).toBe(true);
    });

    it("rejects unsupported versions", () => {
        const json = { version: "1.0", ellipsoid_correction: null, alignment: null, geo_reference: null };
        const result = loadMagCharacterizationModel(json);
        expect(result.valid).toBe(false);
        expect(result.error).toContain("Unsupported model version");
    });

    it("accepts schema 2.0 models", () => {
        const json = {
            version: "2.0",
            ellipsoid_correction: {
                center: { x: 0, y: 0, z: 0 },
                soft_iron: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                radius: 1.0,
                residual_rms: 0.01,
            },
            alignment: { preset: 1 },
            geo_reference: {
                declination_deg: 0,
                inclination_deg: 0,
                field_strength_nt: 50000,
            },
        };

        const result = loadMagCharacterizationModel(json);
        expect(result.valid).toBe(true);
        expect(result.model.version).toBe("2.0");
        expect(result.model.fusion).toBeDefined();
    });
});

describe("computeMagQualityBounds", () => {
    it("accepts a healthy soft-iron matrix", () => {
        const softIron = [
            [0.001, 0, 0],
            [0, 0.001, 0],
            [0, 0, 0.001],
        ];
        const bounds = computeMagQualityBounds(softIron, 50000);
        expect(bounds.field_strength_mg).toBe(500);
        expect(bounds.field_strength_ok).toBe(true);
        expect(bounds.soft_iron_offdiag_ok).toBe(true);
        expect(bounds.soft_iron_anisotropy_ok).toBe(true);
        expect(bounds.bounds_ok).toBe(true);
    });

    it("rejects field strength outside 150-950 mG", () => {
        const softIron = [[0.001, 0, 0], [0, 0.001, 0], [0, 0, 0.001]];
        const weakField = computeMagQualityBounds(softIron, 5000);
        expect(weakField.field_strength_ok).toBe(false);
        expect(weakField.bounds_ok).toBe(false);

        const strongField = computeMagQualityBounds(softIron, 200000);
        expect(strongField.field_strength_ok).toBe(false);
        expect(strongField.bounds_ok).toBe(false);
    });

    it("accepts moderately anisotropic diagonal", () => {
        const softIron = [
            [0.0008, 0, 0],
            [0, 0.0010, 0],
            [0, 0, 0.0009],
        ];
        const bounds = computeMagQualityBounds(softIron, 50000);
        expect(bounds.soft_iron_anisotropy_ok).toBe(true);
    });
});
