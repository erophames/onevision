class AddEnrichmentToPredictionResults < ActiveRecord::Migration[8.0]
  def change
    add_column :prediction_results, :enrichment, :json
  end
end
