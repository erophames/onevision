# syntax=docker/dockerfile:1
# check=error=true

ARG RUBY_VERSION=3.2.3
FROM docker.io/library/ruby:$RUBY_VERSION-slim AS base

# Set workdir
WORKDIR /rails

# Install base packages
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
    curl libjemalloc2 libvips sqlite3 \
    python3.10 python3-pip python3.10-venv tini && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

# Set environment variables
ENV RAILS_ENV="production" \
    BUNDLE_DEPLOYMENT="1" \
    BUNDLE_PATH="/usr/local/bundle" \
    BUNDLE_WITHOUT="development"

# Install Python dependencies
COPY ml/requirements.txt /ml/requirements.txt
RUN python3.10 -m venv /rails/venv && \
    /rails/venv/bin/pip install --no-cache-dir -r /ml/requirements.txt

# Throw-away build stage to reduce image size
FROM base AS build

# Install packages needed to build gems
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y build-essential git pkg-config && \
    rm -rf /var/lib/apt/lists /var/cache/apt/archives

# Install application gems
COPY Gemfile Gemfile.lock ./
RUN bundle install && \
    rm -rf ~/.bundle/ "${BUNDLE_PATH}"/ruby/*/cache "${BUNDLE_PATH}"/ruby/*/bundler/gems/*/.git && \
    bundle exec bootsnap precompile --gemfile

# Copy application code
COPY . .

# Precompile assets & bootsnap cache
RUN bundle exec bootsnap precompile app/ lib/

# Final stage for app image
FROM base

# Copy built artifacts
COPY --from=build "${BUNDLE_PATH}" "${BUNDLE_PATH}"
COPY --from=build /rails /rails

# Copy the ML script and B2 directory
COPY ml/main.py /ml/main.py
COPY ml/B2 /ml/B2

# Run and own only runtime files as a non-root user for security
RUN groupadd --system --gid 1000 rails && \
    useradd rails --uid 1000 --gid 1000 --create-home --shell /bin/bash && \
    chown -R rails:rails db log storage tmp /ml
USER 1000:1000

# Expose ports for Rails (80) and Python (8080)
EXPOSE 80 8080

# Entrypoint prepares the database
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start Rails server, Sidekiq worker, and Python web server
CMD ["sh", "-c", "
  ./bin/thrust ./bin/rails server & \
  bundle exec sidekiq & \
  /rails/venv/bin/python3 /ml/main.py
"]
