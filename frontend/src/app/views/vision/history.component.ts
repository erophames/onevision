import {
  AfterViewInit,
  Component,
  computed,
  Inject,
  OnDestroy,
  OnInit,
  Renderer2,
  signal,
  ViewChild
} from '@angular/core';
import {DatePipe, DOCUMENT, NgForOf, NgStyle} from '@angular/common';

import {
  CardComponent,
  CardBodyComponent,
  RowComponent,
  ColComponent,
  ProgressComponent,
  TableDirective,
  ModalToggleDirective,
  ModalBodyComponent,
  ModalComponent,
  ModalHeaderComponent,
  ButtonDirective,
  WidgetStatFComponent,
  ModalTitleDirective,
  ButtonCloseDirective,
  ModalFooterComponent,
  LocalStorageService
} from '@coreui/angular';
import {IconDirective} from "@coreui/icons-angular";
import {DetectionService} from "../../services/detection.service";
import {cilArrowRight, cilBug, cilChartPie} from "@coreui/icons";
import {Subscription} from "rxjs";
import {getProgressBarColor} from "../../utils/progress-bar-colourr";
import {EnrichmentModalComponent} from "./enrichment/enrichment-modal.component";

@Component({
  standalone: true,
  templateUrl: 'history.component.html',
  imports: [CardComponent, CardBodyComponent, RowComponent, ColComponent, DatePipe, IconDirective, ProgressComponent, TableDirective, NgForOf, ButtonDirective, EnrichmentModalComponent, WidgetStatFComponent, ModalComponent, ModalHeaderComponent, ModalTitleDirective, ButtonCloseDirective, ModalBodyComponent, ModalFooterComponent]
})
export class HistoryComponent implements OnInit, AfterViewInit, OnDestroy {
  protected detections = signal<any[]>([]);
  private detectionSubscription: Subscription | undefined;
  protected isCorruptedEnrichmentVisible = signal<boolean>(false);
  readonly uuid: string | undefined;

  icons = {cilChartPie, cilArrowRight, cilBug};
  meanConfidence = computed(() => {
    const totalConfidence = this.detections().reduce((sum, detection) => sum + parseFloat(detection.confidence), 0);
    return this.detections().length > 0 ? (totalConfidence / this.detections().length).toFixed(2) : '0';
  });
  mostUploadedPlant = computed(() => {
    if (this.detections().length === 0) {
      return 'No plants detected';
    }

    const plantCountMap: { [key: string]: number } = this.detections().reduce((map, detection) => {
      const plant = detection.plant;
      if (plant) {
        // @ts-ignore
        map[plant] = (map[plant] || 0) + 1;
      }
      return map;
    }, {});

    const mostUploaded = Object.keys(plantCountMap).reduce((maxPlant, plant) =>
      plantCountMap[plant] > (plantCountMap[maxPlant] || 0) ? plant : maxPlant, '');

    return mostUploaded || 'No plants detected';
  });

  commonDisease = computed(() => {
    if (this.detections().length === 0) {
      return 'No diseases detected';
    }

    const diseaseCounts: { [key: string]: number } = this.detections().reduce((map, detection) => {
      const disease = detection.disease;
      if (disease) {
        // @ts-ignore
        map[disease] = (map[disease] || 0) + 1;
      }
      return map;
    }, {});

    return Object.keys(diseaseCounts).reduce((maxDisease, disease) =>
      diseaseCounts[disease] > (diseaseCounts[maxDisease] || 0) ? disease : maxDisease, '') || 'No diseases detected';
  });

  @ViewChild(EnrichmentModalComponent) enrichmentModal!: EnrichmentModalComponent;

  constructor(
    @Inject(DOCUMENT) private document: Document,
    private renderer: Renderer2,
    private detectionService: DetectionService,
    private localStorageService: LocalStorageService,
  ) {
    this.uuid = this.localStorageService.getItem('uuid');
  }

  ngOnInit(): void {
    this.detectionSubscription = this.detectionService.getDetections(this.uuid).subscribe((detections: any[]) => {
      const adaptDetections = (detections: any[]): any[] => {
        return detections.sort((a, b) => {
          return (b?.created_at ? new Date(b.created_at).getTime() : 0) - (a?.created_at ? new Date(a.created_at).getTime() : 0);
        }).map(detection => ({
          ...detection,
          confidence: (detection?.confidence ? (Number(detection.confidence) * 100).toFixed(2) : '0.00'),
          progressBarColor: getProgressBarColor(detection?.confidence ? Number((Number(detection.confidence) * 100).toFixed(2)) : 0)
        }));
      };

      this.detections.set(adaptDetections(detections));
    });
  }

  trackByDetectionId(index: number, detection: any): string {
    return detection.detection_id;
  }


  ngAfterViewInit(): void {
  }

  ngOnDestroy(): void {
    if (this.detectionSubscription) {
      {
        this.detectionSubscription.unsubscribe();
      }
    }
    this.detections.set([]);
  }

  openEnrichmentModal(detection: any): void {
    try {
      JSON.parse(detection.enrichment);
      this.enrichmentModal.open(detection);
    } catch {
      this.isCorruptedEnrichmentVisible.update(() => true);
    }
  }

  closeEnrichmentCorruption() {
    this.isCorruptedEnrichmentVisible.set(false);
  }

  handleEnrichmentCorruptionChange(event: any) {
    this.isCorruptedEnrichmentVisible.update(() => event);
  }
}

