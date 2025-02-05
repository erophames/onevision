import {AfterViewInit, Component, input, Input, OnDestroy, OnInit} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ButtonDirective,
  CardBodyComponent,
  CardComponent,
  CardImgDirective,
  ColComponent,
  FormCheckComponent,
  FormCheckInputDirective,
  FormCheckLabelDirective,
  FormControlDirective,
  FormDirective,
  FormLabelDirective,
  FormSelectDirective,
  GutterDirective, ImgDirective,
  ModalBodyComponent,
  ModalComponent, ModalHeaderComponent, ModalTitleDirective,
  RowComponent, TabDirective,
  TabPanelComponent,
  TabsComponent,
  TabsContentComponent,
  TabsListComponent
} from "@coreui/angular";
import {Detection} from "../detection.interface";
import {PlantInterfaceHelper} from "../../../interfaces/plant-interface-helper.interfaces";
import {Disease, Plant} from "../../../interfaces/plant-interfaces";


@Component({
  standalone: true,
  templateUrl: 'plant-info.component.html',
  selector:'app-plant-info',
  styleUrls: ['./plant-info.component.css'],
  imports: [CommonModule, RowComponent, ColComponent, FormControlDirective, CardComponent, CardImgDirective, FormLabelDirective,  GutterDirective, TabsContentComponent, TabPanelComponent, TabsListComponent, TabsComponent, TabDirective, ImgDirective]
})
export class PlantInfoComponent implements OnInit, AfterViewInit, OnDestroy {
  detection = input<PlantInterfaceHelper<Plant, Disease>>();
  constructor(
  ) { }

  ngOnInit(): void {
  }

  ngAfterViewInit(): void { }

  ngOnDestroy(): void {
  }

  protected readonly JSON = JSON;
}
