import {AfterViewInit, Component, input, Input, OnDestroy, OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ColComponent,
  FormControlDirective,
  FormLabelDirective,
  GutterDirective,
  RowComponent, TabDirective,
  TabPanelComponent,
  TabsComponent,
  TabsContentComponent,
  TabsListComponent
} from "@coreui/angular";
import {PlantInterfaceHelper} from "../../../interfaces/plant-interface-helper.interfaces";
import {Disease, Plant} from "../../../interfaces/plant-interfaces";


@Component({
  standalone: true,
  templateUrl: 'disease-info.component.html',
  selector:'app-disease-info',
  styleUrls: ['./disease-info.component.css'],
  imports: [CommonModule, RowComponent, ColComponent, FormControlDirective,  FormLabelDirective,  GutterDirective,
    TabsContentComponent, TabPanelComponent, TabsListComponent, TabsComponent, TabDirective]
})
export class DiseaseInfoComponent {
  detection = input<PlantInterfaceHelper<Plant, Disease>>();
  constructor(
  ) { }
}
