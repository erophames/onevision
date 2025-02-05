class CreateUserCredits < ActiveRecord::Migration[8.0]
  def change
    create_table :user_credits do |t|
      t.integer :user_id, null: false
      t.string :billing_section, null: false
      t.integer :credits, null: false, default: 0

      t.check_constraint "credits >= 0", name: "non_negative_credits"
      t.index [:user_id, :billing_section], unique: true
    end
  end
end

class AddUniqueIndexToUserCredits < ActiveRecord::Migration[7.0]
  def change
    add_index :user_credits, %i[user_id billing_section], unique: true
  end
end
