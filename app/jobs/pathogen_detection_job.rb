# frozen_string_literal: true

# app/jobs/pathogen_detection_job.rb
require 'json'

class PathogenDetectionJob
  include Sidekiq::Job
  sidekiq_options retry: 3, queue: 'critical'

  def initialize
    @logger = Logging.logger[self]

    setup_logging
  end

  def setup_logging
    Logging.logger.root.appenders = Logging.appenders.stdout(
      layout: Logging.layouts.pattern(
        pattern: '[%d] %-5l %c: %m\n',
        date_pattern: '%Y-%m-%d %H:%M:%S'
      )
    )
    @logger.level = ENV['LOG_LEVEL'] || :info
  end

  def perform(image_path, user_id, request_id)
    # Initialize predictor
    # binding.pry
    path = Rails.application.config.pathogen_model_path
    billing = BillingService.new

    begin
      # This user is hard coded so we don't need to setup a million different billing profiles for testing
      # in the real word there would be an entire billing module to control this.
      billing.check_credits_and_execute(1, 'vision', 1) do
        predictor = PathogenPredictorService.new

        result = JSON.parse(predictor.predict(image_path))

        prediction_result = PredictionResult.find_by(request_id: request_id)

        if prediction_result
          prediction_result.update!(status: 'completed', result: result, processed_image: result["processed_image"],
                                    plant_name: result["plant"],disease_name: result["disease"])
        else
          PredictionResult.create!(
            request_id: request_id,
            plant_name: result["plant"],
            disease_name: result["disease"],
            user_id: user_id,
            status: 'completed',
            result: result,
            processed_image: result["processed_image"]
          )

        end
        ActionCable.server.broadcast(
          "pathogen_detection_#{user_id}",
          {
            request_id: request_id,
            status: 'completed',
            processed_at: DateTime.now,
            result: result,
            credits: billing.get_credits(1, 'vision')
          }
        )

        PlantTreatmentEnrichmentJob.perform_async(request_id)

      end
    rescue InsufficientCreditsError => e
      puts "Error: #{e.message}"
    rescue TransactionFailedError => e
      puts "Error: #{e.message}"
    end
  rescue => e
    @logger.error(e)
    raise
  end
end
