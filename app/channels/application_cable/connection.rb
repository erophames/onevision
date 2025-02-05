module ApplicationCable
  class Connection < ActionCable::Connection::Base
    identified_by :current_user_id

    def connect
      self.current_user_id = find_verified_user
    end

    private

    def find_verified_user
      # Retrieve user ID from URL parameters
      user_id = request.params[:user_id]
      if user_id.present?
        user_id
      else
        reject_unauthorized_connection
      end
    end
  end
end