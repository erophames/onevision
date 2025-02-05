class CreatePredictionResults < ActiveRecord::Migration[8.0]
  def change
    create_table :prediction_results do |t|
      t.integer :user_id
      t.string :request_id
      t.integer :status, default: 0  # Ensure default value
      t.text :result

      t.timestamps
    end
  end
end
