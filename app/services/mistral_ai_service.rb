# frozen_string_literal: true

require 'httparty'
require 'json'

class MistralAIService
  include HTTParty
  base_uri 'https://api.mistral.ai/'

  API_KEY ||=  'Ea1loq6zFPOO4HO0ST7367qBG6ZjGwNb'

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

  def call(payload)
    full_payload = {
      model: 'mistral-large-latest',
      messages: [
        {
          role: 'user',
          content: payload
        }
      ]
    }

    @logger.info("Request URL: #{self.class.base_uri}")

    response = self.class.post(
      '/v1/chat/completions',
      body: full_payload.to_json,
      headers: {
        'Content-Type' => 'application/json',
        'Authorization' => "Bearer #{self.class::API_KEY}"
      }
    )

    handle_response(response)
  end

  private

  def handle_response(response)
    if response.success?
      r = JSON.parse(response.body)

      r["choices"][0]["message"]["content"]
    else
      {
        error: response.message,
        code: response.code,
        details: response.body
      }
    end
  end
end