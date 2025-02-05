class PathogenDetectionChannel < ApplicationCable::Channel
  def subscribed
    stream_from "pathogen_detection_#{current_user_id}"
  end

  def unsubscribed
    # Any cleanup needed when channel is unsubscribed
  end
end
