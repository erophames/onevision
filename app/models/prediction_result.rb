class PredictionResult < ApplicationRecord
  enum :status, {
    pending: 0,
    processing: 1,
    completed: 2,
    failed: 3
  }

  validates :request_id, presence: true, uniqueness: true
  validates :user_id, presence: true
  store_accessor :result

  def detection_status_pending?
    status == "pending"
  end

  def self.top_diseases(start_date, end_date, limit = 5)
    where(created_at: start_date..end_date)
      .where.not(disease_name: [nil, ''])
      .group(:disease_name)
      .order(Arel.sql('COUNT(*) DESC'))
      .limit(limit)
      .count
  end

end
