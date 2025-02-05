class CreateBillings < ActiveRecord::Migration[8.0]
  def change
    create_table :billings do |t|
      t.string :key
      t.integer :credit
      t.integer :deduction

      t.timestamps
    end
  end
end
