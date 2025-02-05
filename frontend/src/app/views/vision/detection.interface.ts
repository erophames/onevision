export interface Detection {
  plant: string;
  disease: string;
  confidence: string;
  processed_at: string;
  detection_id: string;
  processed_image: string | null;
  enrichment: string | null;
  progressBarColor: string;
  showEnriched: boolean | null;
  credits_used: string | null;
}
