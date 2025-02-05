import { Plant, Disease, Treatment } from "./plant-interfaces";

type PlantHelperConfig<TPlant, TDisease> = {
  plant: Partial<TPlant>;
  disease: Partial<TDisease>;
  image: string | null;
};

export class PlantInterfaceHelper<
  TPlant extends Record<string, any>,
  TDisease extends Record<string, any>
> {
  public readonly plant: TPlant;
  public readonly disease: TDisease;
  public readonly image: string | undefined;

  constructor(config: PlantHelperConfig<TPlant, TDisease>) {
    if (!config?.plant || !config?.disease) {
      throw new Error("Invalid initialization: Both plant and disease data must be provided");
    }

    this.image = config.image ?? undefined;
    this.plant = this.parsePlantData(config.plant);
    this.disease = this.parseDiseaseData(config.disease);
  }

  private parsePlantData(plantData: Partial<TPlant>): TPlant {
    if (!plantData || typeof plantData !== "object") {
      throw new Error("Invalid plant data: Expected object structure");
    }

    return {
      ...plantData,
      fertilization_strategies: plantData["fertilization_strategies"] ?? "Not provided",
      climatic_conditions: plantData["climatic_conditions"] ?? "Not provided",
      soil_conditions: plantData["soil_conditions"] ?? "Not provided",
      irrigation_practices: plantData["irrigation_practices"] ?? "Not provided",
      harvesting_guidelines: plantData["harvesting_guidelines"] ?? "Not provided",
    } as unknown as TPlant;
  }

  private parseDiseaseData(diseaseData: Partial<TDisease>): TDisease {
    if (!diseaseData || typeof diseaseData !== "object") {
      throw new Error("Invalid disease data: Expected object structure");
    }

    return {
      ...diseaseData,
      treatment: this.parseTreatment(diseaseData["treatment"]),
      IPM_strategies: diseaseData["IPM_strategies"] ?? "Not provided",
    } as unknown as TDisease;
  }

  private parseTreatment(treatment?: Partial<Treatment>): Treatment {
    return {
      organic: treatment?.organic ?? "Not provided",
      chemical: treatment?.chemical ?? "Not provided",
    };
  }
}
