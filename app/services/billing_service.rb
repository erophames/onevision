# frozen_string_literal: true

class InsufficientCreditsError < StandardError; end
class TransactionFailedError < StandardError; end

class BillingService
  MAX_RETRIES = 3

  def check_credits_and_execute(user_id, billing_section, cost_per_call, &block)
    validate_cost(cost_per_call)
    retries = 0

    begin
      UserCredit.transaction do
        # Lock the credit record for atomic updates
        credit = UserCredit.lock.find_or_create_by!(
          user_id: user_id,
          billing_section: billing_section
        ) do |uc|
          uc.credits = 0  # Initialize if new record
        end

        if credit.credits >= cost_per_call
          credit.decrement!(:credits, cost_per_call)
        else
          raise InsufficientCreditsError.new("Insufficient credits for #{billing_section}")
        end
      end

      block.call
    rescue ActiveRecord::RecordNotUnique, ActiveRecord::StaleObjectError
      retries += 1
      retry if retries <= MAX_RETRIES
      raise TransactionFailedError.new("Transaction failed after #{MAX_RETRIES} retries")
    rescue ActiveRecord::LockWaitTimeout
      retries += 1
      if retries <= MAX_RETRIES
        sleep rand(0.1..0.3)
        retry
      else
        raise TransactionFailedError.new("Transaction failed after #{MAX_RETRIES} retries")
      end
    end
  end

  def add_credits(user_id, billing_section, amount)
    validate_amount(amount)
    UserCredit.upsert(
      {
        user_id: user_id,
        billing_section: billing_section,
        credits: amount
      },
      unique_by: %i[user_id billing_section],
      on_duplicate: Arel.sql("credits = user_credits.credits + EXCLUDED.credits")
    )
  end

  def get_credits(user_id, billing_section)
    UserCredit.find_by(user_id: user_id, billing_section: billing_section)&.credits.to_i
  end

  private

  def validate_cost(cost)
    unless cost.is_a?(Integer) && cost.positive?
      raise ArgumentError, "Cost must be positive integer"
    end
  end

  def validate_amount(amount)
    unless amount.is_a?(Integer) && amount >= 0
      raise ArgumentError, "Amount must be non-negative integer"
    end
  end
end