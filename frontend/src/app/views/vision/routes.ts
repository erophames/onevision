import { Routes } from '@angular/router';
import { loadWithRetries } from '../../utils/load-with-retries';

// @ts-ignore
export const routes: Routes = [
  {
    path: '',
    data: {
      title: 'Vision'
    },
    children: [
      {
        path: '',
        redirectTo: 'detect',
        pathMatch: 'full'
      },
      {
        path: 'detect',
        loadComponent: loadWithRetries(() => import('./detect.component').then(m => m.DetectComponent)),
        data: {
          title: 'Detect'
        }
      },
      {
        path: 'history',
        loadComponent: loadWithRetries(() => import('./history.component').then(m => m.HistoryComponent)),
        data: {
          title: 'History'
        }
      },
      {
        path: 'model',
        loadComponent: loadWithRetries(() => import('./model.component').then(m => m.ModelComponent)),
        data: {
          title: 'Model'
        }
      },
    ]
  }
];
