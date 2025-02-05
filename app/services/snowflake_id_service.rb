# frozen_string_literal: true

# Reference: https://en.wikipedia.org/wiki/Snowflake_ID
# Rudimentary implementation to generate id's that have
# low probability of collisions on scaling infrastructure
# - Fabian

class SnowflakeIDService

  EPOCH ||= 157_852_800_000
  MACHINE_ID_BITS ||= 10
  SEQUENCE_BITS ||= 12
  MAX_MACHINE_ID ||= (1 << MACHINE_ID_BITS) - 1
  MAX_SEQUENCE ||= (1 << SEQUENCE_BITS) - 1

  MACHINE_ID_SHIFT ||= SEQUENCE_BITS
  TIMESTAMP_SHIFT ||= MACHINE_ID_SHIFT + MACHINE_ID_BITS
  SEQUENCE_MASK ||= MAX_SEQUENCE

  def initialize(machine_id)
    @machine_id = machine_id
    @last_timestamp = -1
    @sequence = 0

    raise "Machine ID can't be greater than #{MAX_MACHINE_ID}" if @machine_id > MAX_MACHINE_ID
  end

  def generate
    timestamp = current_timestamp

    if timestamp == @last_timestamp
      @sequence = (@sequence + 1) & SEQUENCE_MASK
      raise "Sequence overflow" if @sequence == 0
    else
      @sequence = 0
    end

    @last_timestamp = timestamp

    (timestamp - EPOCH) << TIMESTAMP_SHIFT | (@machine_id << MACHINE_ID_SHIFT) | @sequence
  end

  private

  def current_timestamp
    (Time.now.to_f * 1000).to_i
  end
end

