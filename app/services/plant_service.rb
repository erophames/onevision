require 'json'

class PlantService

  def initialize
    @logger = Logging.logger[self]

    setup_logging
  end

  def get_predictions
    PredictionResult.select(:id, :request_id, :user_id, :created_at, :enrichment, :result).map do |prediction|

      unless prediction.result.nil?
        result = JSON.parse(prediction.result.gsub('=>', ':'))
        {
          id: prediction.id,
          request_id: prediction.request_id,
          user_id: prediction.user_id,
          plant: result["plant"],
          disease: result["disease"],
          confidence: result["confidence"],
          processed_image: result["processed_image"],
          enrichment: prediction.enrichment,
          created_at: prediction.created_at
        }
      end
    end
  end

  private

  def setup_logging
    Logging.logger.root.appenders = Logging.appenders.stdout(
      layout: Logging.layouts.pattern(
        pattern: '[%d] %-5l %c: %m\n',
        date_pattern: '%Y-%m-%d %H:%M:%S'
      )
    )
    @logger.level = ENV['LOG_LEVEL'] || :info
  end

end
