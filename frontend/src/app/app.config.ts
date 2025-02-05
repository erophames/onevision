import { ApplicationConfig, importProvidersFrom } from '@angular/core';
import { provideAnimations } from '@angular/platform-browser/animations';
import {
  provideRouter,
  withEnabledBlockingInitialNavigation,
  withHashLocation,
  withInMemoryScrolling,
  withRouterConfig,
  withViewTransitions
} from '@angular/router';

import { DropdownModule, SidebarModule } from '@coreui/angular';
import { IconSetService } from '@coreui/icons-angular';
import { routes } from './app.routes';
import { ActionCableService } from './services/action-cable.service';
import {FileUploadService} from "./services/file-upload.service";
import {provideHttpClient} from "@angular/common/http";
import {SnowflakeIdServiceConfig} from "./services/snowflake-id.service";

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes,
      withRouterConfig({
        onSameUrlNavigation: 'reload'
      }),
      withInMemoryScrolling({
        scrollPositionRestoration: 'top',
        anchorScrolling: 'enabled'
      }),
      withEnabledBlockingInitialNavigation(),
      withViewTransitions(),
      withHashLocation()
    ),
    importProvidersFrom(SidebarModule, DropdownModule),
    IconSetService,
    provideAnimations(),
    provideHttpClient(),
    ActionCableService,
    FileUploadService,
    {
      provide: 'SnowflakeIdServiceConfig',
      useValue: {
        workerId: BigInt(1),
        datacenterId: BigInt(1),
      } as SnowflakeIdServiceConfig
    }
  ]
};
