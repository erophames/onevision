import { Injectable, Inject } from '@angular/core';
export interface SnowflakeIdServiceConfig {
  workerId: bigint;
  datacenterId: bigint;
}
@Injectable({
  providedIn: 'root',
})
export class SnowflakeIdService {
  private static readonly EPOCH: bigint = BigInt(1704067200000);
  private static readonly WORKER_ID_BITS = 5n;
  private static readonly DATACENTER_ID_BITS = 5n;
  private static readonly SEQUENCE_BITS = 12n;

  private static readonly MAX_WORKER_ID = (1n << SnowflakeIdService.WORKER_ID_BITS) - 1n;
  private static readonly MAX_DATACENTER_ID = (1n << SnowflakeIdService.DATACENTER_ID_BITS) - 1n;
  private static readonly MAX_SEQUENCE = (1n << SnowflakeIdService.SEQUENCE_BITS) - 1n;

  private static readonly WORKER_ID_SHIFT = SnowflakeIdService.SEQUENCE_BITS;
  private static readonly DATACENTER_ID_SHIFT = SnowflakeIdService.SEQUENCE_BITS + SnowflakeIdService.WORKER_ID_BITS;
  private static readonly TIMESTAMP_SHIFT = SnowflakeIdService.SEQUENCE_BITS + SnowflakeIdService.WORKER_ID_BITS + SnowflakeIdService.DATACENTER_ID_BITS;

  private lastTimestamp: bigint = -1n;
  private sequence: bigint = 0n;

  private workerId: bigint;
  private datacenterId: bigint;

  constructor(
    @Inject('SnowflakeIdServiceConfig') private readonly config: SnowflakeIdServiceConfig
  ) {
    const { workerId, datacenterId } = config;

    if (workerId > SnowflakeIdService.MAX_WORKER_ID || workerId < 0n) {
      throw new Error(`workerId must be between 0 and ${SnowflakeIdService.MAX_WORKER_ID}`);
    }
    if (datacenterId > SnowflakeIdService.MAX_DATACENTER_ID || datacenterId < 0n) {
      throw new Error(`datacenterId must be between 0 and ${SnowflakeIdService.MAX_DATACENTER_ID}`);
    }

    this.workerId = workerId;
    this.datacenterId = datacenterId;
  }

  private currentTimestamp(): bigint {
    return BigInt(Date.now());
  }

  private waitNextMillis(lastTimestamp: bigint): bigint {
    let timestamp = this.currentTimestamp();
    while (timestamp <= lastTimestamp) {
      timestamp = this.currentTimestamp();
    }
    return timestamp;
  }

  public generate(): bigint {
    let timestamp = this.currentTimestamp();

    if (timestamp < this.lastTimestamp) {
      throw new Error('Clock moved backwards. Refusing to generate ID.');
    }

    if (timestamp === this.lastTimestamp) {
      this.sequence = (this.sequence + 1n) & SnowflakeIdService.MAX_SEQUENCE;
      if (this.sequence === 0n) {
        timestamp = this.waitNextMillis(this.lastTimestamp);
      }
    } else {
      this.sequence = 0n;
    }

    this.lastTimestamp = timestamp;

    return (
      ((timestamp - SnowflakeIdService.EPOCH) << SnowflakeIdService.TIMESTAMP_SHIFT) |
      (this.datacenterId << SnowflakeIdService.DATACENTER_ID_SHIFT) |
      (this.workerId << SnowflakeIdService.WORKER_ID_SHIFT) |
      this.sequence
    );
  }
}
