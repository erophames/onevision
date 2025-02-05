# frozen_string_literal: true
require_relative '../services/mistral_ai_service'
require 'erb'

class PlantTreatmentEnrichmentJob
  include Sidekiq::Job
  sidekiq_options retry: 3, queue: "critical"
  PROMPT = "Provide a detailed JSON response for the treatment of a specified plant disease: <%= plant %> affected by <%= disease %> in <%= country %>.
            The JSON response must strictly follow the structure provided below without adding or removing any properties. Do NOT include new line characters,
            carriage returns, or additional formatting—return only the raw JSON structure. The response should include both organic and chemical treatment methods.
            Organic treatments should focus on natural and sustainable approaches such as crop rotation, resistant varieties, organic sprays, and biological control methods.
            Chemical treatments should specify the chemical names, application methods, preventive or curative properties, and recommended dosages if available.
            Include pest- or disease-resistant varieties where applicable. Ensure that fertilization strategies, climatic conditions, soil conditions, irrigation practices,
            and harvesting guidelines are populated with relevant information regarding <%= plant %>. Additionally, provide detailed information on the disease,
            including symptoms, available organic and chemical treatments, the latest Integrated Pest Management (IPM) strategies, the economic impact of <%= plant %>
            affected by <%= disease %>, regional specifics for <%= country %>, monitoring guidelines, recent research developments, and sustainable practices.
            The JSON structure is frozen, and no additional properties are to be added within the defined structure..
            The JSON must follow this format: {\"plant\": {\"latin_name\": \"\", \"name\": \"\", \"information\": \"\", \"pest_resistant_varieties\": \"\",
            \"fertilization_strategies\": \"\", \"climatic_conditions\": \"\", \"soil_conditions\": \"\", \"irrigation_practices\": \"\", \"harvesting_guidelines\": \"\"},
            \"disease\": {\"name\": \"\", \"information\": \"\", \"symptoms\": \"\", \"treatment\": {\"organic\": \"\", \"chemical\": \"\"}, \"IPM_strategies\": \"\",
            \"economic_impact\": \"\", \"regional_specifics\": \"\", \"monitoring_guidelines\": \"\", \"research_developments\": \"\", \"sustainable_practices\": \"\"}}."


  # PROMPT = "Provide a detailed JSON response for the treatment of a specified plant disease - <%= plant %>  <%= disease %> in the region of <%= country %>.
  #           The response should include both organic and chemical treatment methods. The organic treatments should focus on natural or sustainable approaches,
  #           such as crop rotation, resistant varieties, and organic sprays. The chemical treatments should specify the chemical names and describe their methods of application,
  #           including any preventive or curative properties. Ensure the response is concise, organized, and does not include any explanations or confirmation responses.
  #           The JSON should not have any new line carriage returns or additional formatting\u2014provide only the raw JSON structure.
  #            It must always be returned in the following structure. Include resistant varieties.
  #           \r\n\r\n\r\n{\r\n  \"plant\": {\r\n  \"latin_name\":\"\",  \"name\": \"\",\r\n    \"information\": \"\",\r\n    \"pest_resistant_varieties\": [],\r\n
  #           \"fertilization_strategies\": {},\r\n    \"climatic_conditions\": {},\r\n    \"soil_conditions\": {},\r\n    \"irrigation_practices\": {},\r\n
  #           \"harvesting_guidelines\": {}\r\n  },\r\n  \"disease\": {\r\n    \"name\": \"\",\r\n    \"information\": \"\",\r\n    \"symptoms\": [],\r\n
  #           \"treatment\": {\r\n      \"organic\": [],\r\n      \"chemical\": []\r\n    },\r\n    \"IPM_strategies\": {},\r\n    \"economic_impact\": \"\",\r\n
  #           \"government_initiatives\": \"\",\r\n    \"regional_specifics\": \"\",\r\n    \"monitoring_guidelines\": \"\",\r\n    \"research_developments\": \"\",\r\n
  #           \"sustainable_practices\": \"\"\r\n  }\r\n}\r\nProvide the information regarding the plant in this case <%= plant %>, give informative information regarding this crop,
  #           pest resistance varieties to the disease mentioned above. Then include information about the disease in question as a property to the primary JSON result,
  #           giving as much information about the disease in question. Include best fertilization strategies for the crop above.\r\n\r\nInclude information about the ideal
  #           climatic conditions for growing <%= plant %> in <%= country %> and describe how different climatic conditions can affect the prevalence of <%= plant %>  <%= disease %>.
  #           Provide details on the soil types and conditions that are best suited for <%= plant %> cultivation and mention any soil amendments or preparations that can help in preventing <%= plant %>  <%= disease %>.
  #           Include best practices for irrigation to ensure optimal growth and reduce the risk of disease and describe how proper irrigation can help in managing <%= plant %>  <%= disease %>.
  #           Provide guidelines for harvesting <%= plant %> to minimize the spread of <%= plant %>  <%= disease %> and include post-harvest management practices to prevent the disease from
  #           affecting stored product.\r\n\r\nDescribe IPM strategies that combine biological, cultural, and chemical methods to control <%= plant %>  <%= disease %> and include information
  #           on beneficial insects or microorganisms that can help control the disease. Provide information on the economic impact of <%= plant %>  <%= disease %> on <%= plant %> production
  #           in <%= country %> and include any government or industry initiatives aimed at controlling the disease. Mention any regional specifics or local practices that are
  #           particularly effective in managing <%= plant %> <%= disease %> in <%= country %> and include information on local varieties or practices that have shown success in disease
  #           management.\r\n\r\nProvide guidelines for monitoring <%= plant %>  fields for early signs of <%= plant %> <%= disease %> and include information on tools or technologies
  #           that can aid in early detection and management of the disease. Mention any ongoing research or developments in the field of <%= plant %>  <%= disease %> management and
  #          include information on new varieties or treatments that are being developed or tested. Provide information on sustainable farming practices that can help in
  #          long-term management of <%= plant %>  <%= disease %> and include details on organic farming methods and their benefits in disease control. You are NOT to return the result with any JSON markdown formatting."

  def initialize
    @logger = Logging.logger[self]
    setup_logging
  end

  def setup_logging
    Logging.logger.root.appenders = Logging.appenders.stdout(
      layout: Logging.layouts.pattern(
        pattern: '[%d] %-5l %c: %m\n',
        date_pattern: "%Y-%m-%d %H:%M:%S"
      )
    )
    @logger.level = ENV["LOG_LEVEL"] || :info
  end

  def perform(request_id)
    prediction_result = PredictionResult.find_by(request_id: request_id)

    if prediction_result.nil?
      @logger.error("No prediction result found for request_id: #{request_id}")
      return
    end

    if prediction_result.result.present?
        json = JSON.parse(prediction_result.result.gsub('=>', ':'))


        plant = json['plant']
        disease = json['disease']
        country = 'South Africa'

        if disease.include?("healthy")
          @logger.info("Ignoring #{request_id} as detection was healthy")

          return
        end

        renderer = ERB.new(PROMPT)
      begin
        service = MistralAIService.new
        enrichment = service.call(renderer.result(binding))

        if enrichment.present?
          prediction_result.update!(enrichment: enrichment)

          ActionCable.server.broadcast(
            "pathogen_enrichment_#{prediction_result.user_id}",
            {
              request_id: request_id,
              enrichment: enrichment
            }
          )
        end

      rescue => e
        @logger.error("Error while calling MistralAIService: #{e.message}")
        raise
      end
    else
      @logger.error("Prediction result is empty for request_id: #{request_id}")
    end
  rescue => e
    @logger.error("Unexpected error: #{e.message}")
    raise
  end
end
