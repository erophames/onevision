export interface Plant {
  readonly latin_name: string;
  readonly name: string;
  readonly information: string;
  readonly pest_resistant_varieties: string;
  readonly fertilization_strategies: string;
  readonly climatic_conditions: string;
  readonly soil_conditions: string;
  readonly irrigation_practices: string;
  readonly harvesting_guidelines: string;
}

export interface Treatment {
  readonly organic: string;
  readonly chemical: string;
}

export interface Disease {
  readonly name: string;
  readonly information: string;
  readonly symptoms: string;
  readonly treatment: Treatment;
  readonly IPM_strategies: string;
  readonly economic_impact: string;
  readonly regional_specifics: string;
  readonly monitoring_guidelines: string;
  readonly research_developments: string;
  readonly sustainable_practices: string;
}

export interface PlantDiseaseData {
  readonly plant: Plant;
  readonly disease: Disease;
  readonly image: string;
}
