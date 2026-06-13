import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, resolve } from "path";
import { loadMagCharacterizationModel } from "../../src/mag_model.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe("flight1 model 2.2 — integration smoke test", () => {
    it("loads and validates the flight1 model_2.2.json", () => {
        const modelPath = resolve(
            __dirname,
            "..",
            "..",
            "..",
            "planv5",
            "blackbox",
            "flight1",
            "model_2.2.json",
        );
        const json = JSON.parse(readFileSync(modelPath, "utf-8"));

        const result = loadMagCharacterizationModel(json);
        expect(result.valid).toBe(true);
        expect(result.model.version).toBe("2.2");
        expect(result.model.fusion).toBeDefined();
        expect(result.model.fusion.frame).toBe("FRD");
        expect(result.model.fusion.gaussPerCorrectedUnit).toBeGreaterThan(0);
        expect(result.model.fusion.earthFieldNedGauss).toBeTruthy();
        expect(result.model.fusion.magNoiseGauss.sigma).toBeGreaterThan(0);
        expect(result.model.fusion.qualityBounds.bounds_ok).toBe(true);
    });
});
