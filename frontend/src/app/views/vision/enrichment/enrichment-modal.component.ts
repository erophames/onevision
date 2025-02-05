import { AfterViewInit, Component, ElementRef, Input, OnInit, ViewChild } from '@angular/core';
import {
  ButtonCloseDirective,
  ModalBodyComponent,
  ModalComponent,
  ModalHeaderComponent, ModalTitleDirective, TabDirective, TabPanelComponent,
  TabsComponent, TabsContentComponent,
  TabsListComponent
} from "@coreui/angular";
import { PlantInfoComponent } from "./plant-info.component";
import { Detection } from "../detection.interface";
import { Disease, Plant } from "../../../interfaces/plant-interfaces";
import { PlantInterfaceHelper } from "../../../interfaces/plant-interface-helper.interfaces";
import {MimeTypeService} from "../../../services/mine-type.service";
import {DiseaseInfoComponent} from "./disease-info.component";

@Component({
  standalone: true,
  selector: 'app-enrichment-modal',
  templateUrl: 'enrichment-modal.component.html',
  styleUrls: ['enrichment-modal.component.css'],
  imports: [
    ModalComponent,
    ModalHeaderComponent,
    ModalBodyComponent,
    TabsComponent,
    TabsListComponent,
    TabsContentComponent,
    TabPanelComponent,
    PlantInfoComponent,
    ModalTitleDirective,
    TabDirective,
    DiseaseInfoComponent,
    ButtonCloseDirective
  ]
})
export class EnrichmentModalComponent {
  @ViewChild(ModalComponent) modal!: ModalComponent;
  detection!: PlantInterfaceHelper<Plant, Disease>;

  constructor(private el: ElementRef, private mimeTypeService: MimeTypeService) {}

  get element(): HTMLElement {
    return this.el.nativeElement;
  }

  close() {
    this.modal.visible = false;
  }

  open(input: Detection) {
    if (typeof input.enrichment === "string") {
      this.detection = new PlantInterfaceHelper<Plant, Disease>({...JSON.parse(input.enrichment),
        image: this.mimeTypeService.detectMimeType(input.processed_image)});
      this.modal.visible = true;
    } else if (typeof input.enrichment === "object") {
      this.detection = new PlantInterfaceHelper<Plant, Disease>({...input.enrichment as any,
        image: this.mimeTypeService.detectMimeType(input.processed_image)});
      this.modal.visible = true;
    }
  }
}
