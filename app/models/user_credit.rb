class UserCredit < ApplicationRecord
  validates :user_id, presence: true
  validates :billing_section, presence: true
  validates :credits, numericality: { greater_than_or_equal_to: 0 }
  validates :billing_section, uniqueness: { scope: :user_id }
end
