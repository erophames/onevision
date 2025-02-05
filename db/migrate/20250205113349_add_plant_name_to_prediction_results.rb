class AddPlantNameToPredictionResults < ActiveRecord::Migration[8.0]
  def change
    add_column :prediction_results, :plant_name, :string, null: false, default: ''
  end
end
