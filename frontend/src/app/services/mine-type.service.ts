import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root',
})
export class MimeTypeService {
  constructor() {}

  detectMimeType(base64: string | null): string | null {
    if (base64) {
      if (base64.startsWith('data:image/png;base64,')) return base64;
      if (base64.startsWith('data:image/jpeg;base64,')) return base64;
      if (base64.startsWith('data:image/gif;base64,')) return base64;

      if (base64.startsWith('iVBORw0K')) return `data:image/png;base64,${base64}`;
      if (base64.startsWith('/9j/')) return `data:image/jpeg;base64,${base64}`;
      if (base64.startsWith('R0lGODlh')) return `data:image/gif;base64,${base64}`;

      return `data:image/png;base64,${base64}`; // Default to PNG if not recognized
    } else {
      return null;
    }
  }
}
