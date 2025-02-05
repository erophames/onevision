class AddProcessedImageToYourModel < ActiveRecord::Migration[8.0]
  def change
    add_column :prediction_results, :processed_image, :text
  end
end
