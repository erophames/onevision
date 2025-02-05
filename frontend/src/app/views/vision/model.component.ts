import {
  Component,
  Inject,
} from '@angular/core';
import {DOCUMENT} from '@angular/common';


@Component({
  standalone: true,
  templateUrl: 'model.component.html',
  styleUrls: ['model.component.css'],
  imports: []
})
export class ModelComponent {

  constructor(
    @Inject(DOCUMENT) private document: Document
  ) {
  }
}

