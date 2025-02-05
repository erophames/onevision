import { INavData } from '@coreui/angular';

export const navItems: INavData[] = [
  {
    name: 'Dashboard',
    url: '/dashboard',
    iconComponent: { name: 'cil-speedometer' },
  },
  {
    title: true,
    name: 'Vision'
  },
  {
    name: 'Detect',
    url: '/vision/detect',
    iconComponent: { name: 'cil-magnifying-glass' }
  },
  {
    name: 'History',
    url: '/vision/history',
    iconComponent: { name: 'cil-history' }
  },
  {
    name: 'Model',
    url: '/vision/model',
    iconComponent: { name: 'cil-vector' }
  }
];
