require 'sidekiq/web'

Rails.application.routes.draw do
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Reveal health status on /up that returns 200 if the app boots with no exceptions, otherwise 500.
  # Can be used by load balancers and uptime monitors to verify that the app is live.
  get "up" => "rails/health#show", as: :rails_health_check

  # Defines the root path route ("/")
  # root "posts#index"

  resources :detections, only: [:create]
  get '/detections', to: 'detections#index'
  get 'detections/status/:detection_id', to: 'detections#status', as: 'status'
  get 'detections/dashboard', to: 'detections#dashboard', as: 'dashboard'

  mount Sidekiq::Web => '/sidekiq'
  mount ActionCable.server => "/cable"
end
