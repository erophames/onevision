class AddDiseaseNameToPredictionResults < ActiveRecord::Migration[8.0]
  def change
    add_column :prediction_results, :disease_name, :string
  end
end
