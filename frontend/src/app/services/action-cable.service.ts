import { Injectable } from '@angular/core';
import { ReplaySubject, Observable } from 'rxjs';
import * as ActionCable from 'actioncable';

interface ChannelConfig {
  channel: string;
  userId?: string;
  params?: any;
  onConnected?: () => void;
  onDisconnected?: () => void;
  onReceived?: (data: any) => void;
}

@Injectable({
  providedIn: 'root',
})
export class ActionCableService {
  private cable: any;
  private subscriptions: Map<string, any> = new Map();
  private messageStreams: Map<string, ReplaySubject<any>> = new Map();

  constructor() {
  }

  initializeConnection(userId: string): void {
    if (this.cable) {
      this.cable.disconnect(); // Disconnect existing connection if any
    }
    // Include user_id in the WebSocket URL
    this.cable = ActionCable.createConsumer(`wss://cable.rednode.co.za/cable?user_id=${userId}`);
  }
  private createChannelSubscription(config: ChannelConfig) {
    if (!this.cable) {
      throw new Error('ActionCable connection not initialized. Call initializeConnection() first.');
    }
    const messageStream = new ReplaySubject<any>(1);
    const subscription = this.cable.subscriptions.create(
      { channel: config.channel, user_id: config.userId, ...config.params },
      {
        connected: () => {
          console.log(`${config.channel} connected`);
          config.onConnected?.();
        },
        disconnected: () => {
          console.log(`${config.channel} disconnected`);
          config.onDisconnected?.();
        },
        received: (data: any) => {
          console.log(`Received data on ${config.channel}:`, data);
          if (typeof config.onReceived === 'function') {
            config.onReceived(data);
          } else {
            console.warn(`No 'onReceived' callback defined for channel: ${config.channel}`);
          }

          messageStream.next(data);
        },
        rejected: (error: any) => {
          console.error(`${config.channel} subscription rejected:`, error);
        },
      }
    );
    this.subscriptions.set(config.channel, subscription);
    this.messageStreams.set(config.channel, messageStream);
    return messageStream;
  }

  subscribeToChannel(config: ChannelConfig): Observable<any> {
    if (!this.messageStreams.has(config.channel)) {
      this.createChannelSubscription(config);
    }
    return this.messageStreams.get(config.channel)!.asObservable();
  }

  unsubscribeFromChannel(channelName: string): void {
    const subscription = this.subscriptions.get(channelName);
    if (subscription) {
      this.cable.subscriptions.remove(subscription);
      this.subscriptions.delete(channelName);
      this.messageStreams.delete(channelName);
      console.log(`'❌' Unsubscribed from ${channelName}`);
    }
  }

  sendMessage(channelName: string, event: string, payload: any): void {
    const subscription = this.subscriptions.get(channelName);
    if (subscription) {
      subscription.perform(event, payload);
    }
  }

  cleanup(): void {
    this.cable.subscriptions.subscriptions.forEach((subscription: any) => {
      this.cable.subscriptions.remove(subscription);
    });
    this.subscriptions.clear();
    this.messageStreams.clear();
    console.log('All subscriptions removed');
  }
}
