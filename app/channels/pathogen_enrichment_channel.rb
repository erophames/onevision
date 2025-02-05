# frozen_string_literal: true
class PathogenEnrichmentChannel < ApplicationCable::Channel
  def subscribed
    stream_from "pathogen_enrichment_#{current_user_id}"
  end

  def unsubscribed
    # Any cleanup needed when channel is unsubscribed
  end
end

