# frozen_string_literal: true

Rails.application.config.content_security_policy do |policy|
  policy.connect_src :self, "wss://cable.rednode.co.za"
end
