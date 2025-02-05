# app/controllers/detections_controller.rb
require_relative '../services/snowflake_id_service'

class DetectionsController < ApplicationController
  def index
    user_id = params[:user_id]

    begin
      predictions = PredictionResult
                      .where(user_id: user_id)
                      .select(:id, :request_id, :user_id, :created_at, :enrichment, :result)
                      .map do |prediction|

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

      render json: predictions.compact
    rescue => e
      render json: { error: "An error has occurred: #{e.message}" }, status: 500
    end
  end

  def dashboard
    start_date = params[:start_date] || 30.days.ago
    end_date = params[:end_date] || Time.now

    top_diseases = PredictionResult.top_diseases(start_date, end_date)

    render json: top_diseases
  end

  def create
    uploaded_file = params[:image]
    user_id = params[:user_id]

    if uploaded_file.nil?
      return render json: { error: 'No file uploaded' }, status: :bad_request
    end

    unless uploaded_file.content_type.start_with?('image/')
      return render json: { error: 'Invalid file type. Please upload an image.' }, status: :bad_request
    end

    file_extension = File.extname(uploaded_file.original_filename)
    storage_path = Rails.root.join('storage', 'uploads', "#{SecureRandom.uuid}#{file_extension}")

    begin
      FileUtils.mv(uploaded_file.tempfile, storage_path)
    rescue => e
      return render json: { error: "Failed to save file: #{e.message}" }, status: :internal_server_error
    end

    snowflakeService = SnowflakeIDService.new(1)

    prediction = PredictionResult.create!(
      user_id: user_id,
      request_id: snowflakeService.generate(),
      status: :pending
    )

    begin
      PathogenDetectionJob.perform_async(
        storage_path.to_s,
        user_id,
        prediction.request_id
      )
    rescue => e
      prediction.update!(status: :failed, error_message: { message: "Job failed: #{e.message}" })
      return render json: { error: "Failed to process image: #{e.message}" }, status: :internal_server_error
    end

    render json: {
      detection_id: prediction.request_id,
      status_id: prediction.status,
      user_id: user_id,
      status_url: url_for(controller: 'detections', action: 'status', detection_id: prediction.request_id, only_path: false)
    }, status: :accepted
  end

  def status
    prediction = PredictionResult.find_by(request_id: params[:detection_id])

    if prediction.nil?
      return render json: { error: "Prediction result not found" }, status: :not_found
    end

    response = {
      detection_id: prediction.request_id,
      status_id: prediction.status,
      created_at: prediction.created_at
    }

    if prediction.completed?
      response.merge!(
        result: prediction.result,
        processed_at: prediction.updated_at
      )
    elsif prediction.failed?
      response[:error] = prediction.result
    end

    render json: response
  end
end
