require 'httparty'
require 'json'
require 'logging'
require 'concurrent'

class PathogenPredictorService
  include HTTParty
  # Configuration with environment variables
  class Config
    attr_accessor :img_size, :tta_steps, :cache_ttl, :max_batch_size, :rate_limit, :api_url

    def initialize
      @img_size = ENV.fetch('IMG_SIZE', 300).to_i
      @tta_steps = ENV.fetch('TTA_STEPS', 5).to_i
      @cache_ttl = ENV.fetch('CACHE_TTL', 3600).to_i
      @max_batch_size = ENV.fetch('MAX_BATCH_SIZE', 32).to_i
      @rate_limit = ENV.fetch('RATE_LIMIT', 100).to_i
      @api_url = ENV.fetch('API_URL', 'http://127.0.0.1:8080/predict') # Default URL for FastAPI service
    end
  end

  # Custom Exceptions
  class PredictionError < StandardError; end

  class InvalidInputError < PredictionError; end

  def initialize
    @config = Config.new
    @logger = Logging.logger[self]
    @cache = Concurrent::Map.new
    @semaphore = Concurrent::Semaphore.new(@config.rate_limit)

    setup_logging
  end

  def predict(image_path)
    @semaphore.acquire
    start_time = Time.now

    validate_input!(image_path)
    result = process_prediction(image_path)

    result.to_json
  rescue PredictionError => e
    @logger.error "Prediction failed: #{e.message}", backtrace: e.backtrace
    raise
  ensure
    @semaphore.release
  end

  def healthcheck
    {
      status: :ok,
      cache_size: @cache.size,
      uptime: Time.now - @start_time
    }
  rescue => e
    { status: :error, message: e.message }
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

  def validate_input!(image_path)
    raise InvalidInputError, "File not found: #{image_path}" unless File.exist?(image_path)
    mime_type = MIME::Types.type_for(image_path).first&.content_type
    valid_types = ['image/jpeg', 'image/png', 'image/tiff']
    raise InvalidInputError, "Invalid file type: #{mime_type || 'unknown'}" unless valid_types.include?(mime_type)
    raise InvalidInputError, "File size exceeds limit (10MB)" if File.size(image_path) > 10_000_000
  end

  def process_prediction(image_path)
    begin
      cached = @cache[image_path]
      return cached[:result] if cached && (Time.now - cached[:timestamp] < @config.cache_ttl)

      # Read image and send it to FastAPI service for prediction
      image_bytes = File.read(image_path)
      response = self.class.post(
        @config.api_url,
        body: {
          file: File.new(image_path) # Sending the image as a file
        },
        headers: { 'Content-Type' => 'multipart/form-data' }
      )

      if response.success?
        result = JSON.parse(response.body)
        # Cache the result
        @cache.compute_if_absent(image_path) { { result: result, timestamp: Time.now } }
        result
      else
        raise PredictionError, "Prediction failed: #{response.message}"
      end
    rescue StandardError => e
      # Handle any exception that might occur during the process
      raise PredictionError, "An error occurred during prediction: #{e.message}"
    end
  end
end