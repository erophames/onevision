import {
  AfterViewInit, ChangeDetectionStrategy, ChangeDetectorRef,
  Component,
  ComponentRef, computed,
  Inject,
  OnDestroy,
  OnInit,
  Renderer2,
  signal,
  ViewChild
} from '@angular/core';
import { CommonModule, DatePipe, DOCUMENT } from '@angular/common';
import {cilArrowRight, cilBug, cilChartPie} from '@coreui/icons';
import {
  CardComponent,
  CardBodyComponent,
  RowComponent,
  ColComponent,
  ProgressComponent,
  TableDirective,
  ImgDirective,
  FormControlDirective,
  WidgetStatFComponent,
  AlertComponent,
  LocalStorageService,
  ToasterPlacement,
  ButtonDirective,
  ButtonCloseDirective,
  ModalBodyComponent,
  ModalComponent,
  ModalFooterComponent, ModalHeaderComponent, ModalTitleDirective
} from '@coreui/angular';
import { ActionCableService } from '../../services/action-cable.service';
import { FileUploadService } from '../../services/file-upload.service';
import {Subscription} from "rxjs";
import {EnrichmentModalComponent} from "./enrichment/enrichment-modal.component";
import {Detection} from "./detection.interface";
import {MimeTypeService} from "../../services/mine-type.service";

@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  templateUrl: 'detect.component.html',
  styleUrls: ['detect.component.css'],
  imports: [CommonModule, ImgDirective, FormControlDirective, WidgetStatFComponent, CardComponent, CardBodyComponent, RowComponent, ColComponent, ProgressComponent, TableDirective, DatePipe, AlertComponent, EnrichmentModalComponent, ButtonDirective, ButtonCloseDirective, ModalBodyComponent, ModalComponent, ModalFooterComponent, ModalHeaderComponent, ModalTitleDirective]
})
export class DetectComponent implements OnInit, AfterViewInit, OnDestroy {

  protected pathogenChannelSubscription: Subscription | undefined;
  protected pathogenEnrichmentSubscription: Subscription | undefined;
  protected isCorruptedEnrichmentVisible = signal<boolean>(false);
  currentPlant = signal<string>('');
  uploadProgress = signal<number>(0);
  isUploading = signal<boolean>(false);
  hasCredits = signal<boolean>(true);
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
  visible = signal<boolean>(false);
  credits = signal<number>(0);
  detections =  signal<Detection[]>([]);

  icons = { cilChartPie, cilArrowRight, cilBug };
  selectedFile: File | null = null;
  img: string | null ='assets/images/empty_img.png';
  placement = ToasterPlacement.TopCenter

  @ViewChild(EnrichmentModalComponent) enrichmentModal!: EnrichmentModalComponent;
  constructor(
    @Inject(DOCUMENT) private document: Document,
    private actionCableService: ActionCableService,
    private localStorageService: LocalStorageService,
    private fileUploadService: FileUploadService,
    private cdr: ChangeDetectorRef,
    private mimeTypeService: MimeTypeService,
  ) { }

  ngOnInit(): void {
    this.initializeCredits();
    this.subscribeToDetectionData();
  }

  ngAfterViewInit(): void { }

  ngOnDestroy(): void {
    this.resetSignals();
    this.unsubscribeFromChannels();
  }

  private initializeCredits(): void {
    const credits = this.localStorageService.getItem('credits');
    if (credits !== null) {
      this.credits.set(credits);
      this.hasCredits.set(true);
    }
  }

  private subscribeToChannel(channelName: string, onReceived: (data: any) => void): Subscription {
    return this.actionCableService
      .subscribeToChannel({
        channel: channelName,
        onConnected: () => {
          console.log(`✅ Connected to ${channelName}`);
        },
        onReceived: (data: any) => onReceived(data),
        onDisconnected: () => {
          console.log(`❌ Disconnected from ${channelName}`);
        },
      })
      .subscribe();
  }
  private subscribeToDetectionData(): void {
    console.info('Subscribing to channels...');
    this.pathogenChannelSubscription = this.subscribeToChannel(
      'PathogenDetectionChannel',
      (data) => this.handleDetectionData(data)
    );

    this.pathogenEnrichmentSubscription = this.subscribeToChannel(
      'PathogenEnrichmentChannel',
      (data) => this.handleEnrichmentData(data)
    );
  }


  private handleEnrichmentData(data: any): void {
    const enrichment = JSON.parse(data.enrichment);

    if (enrichment && data.request_id) {

      this.detections.update((currentDetections) =>
        currentDetections.map((detection) =>
          detection.detection_id ===  data.request_id
            ? { ...detection, enrichment, showEnriched: true }
            : detection
        )
      );
      setTimeout(() => {
        this.detections.update((currentDetections) =>
          currentDetections.map((detection) =>
            detection.detection_id ===  data.request_id
              ? { ...detection, showEnriched: false }
              : detection
          )
        );
      }, 3000);
    }
  }
  private resetSignals(): void {
    this.uploadProgress.set(0);
    this.isUploading.set(false);
    this.hasCredits.set(true);
    this.visible.set(false);
    this.credits.set(0);
    this.detections.set([]);
    this.img = 'assets/images/empty_img.png';
  }
  private handleDetectionData(data: any): void {
    if (data.result) {
      this.img = this.mimeTypeService.detectMimeType(data.result.processed_image);
    }

    const { request_id, processed_at, result } = data;
    const newDetection = this.createDetection(request_id, processed_at, result);
    this.detections.update((currentDetections) => [...currentDetections, newDetection]);

    this.localStorageService.setItem('credits', data.credits);

    this.currentPlant.set(`${newDetection.plant} ${newDetection.disease}`);
    this.hasCredits.update(() => data.credits > 0);
    this.credits.update(() => data.credits);

    this.detections.update((currentDetections) =>
      [...currentDetections].sort(
        (a, b) => new Date(b.processed_at).getTime() - new Date(a.processed_at).getTime()
      )
    );

    this.uploadProgress.set(0);

    this.cdr.detectChanges();
  }

  private createDetection(request_id: string, processed_at: string, result: any): Detection {
    return {
      plant: result.plant || 'Unknown',
      disease: result.disease || 'Unknown',
      confidence: (result.confidence * 100).toFixed(2),
      processed_at,
      detection_id: request_id,
      enrichment: null,
      showEnriched: null,
      credits_used: result.credits_used || 0,
      processed_image: null,
      progressBarColor: this.getProgressBarColor(result.confidence * 100)
    };
  }

  private getProgressBarColor(percent: number): string {
    if (percent >= 75) return 'success';
    if (percent >= 50) return 'warning';
    if (percent >= 25) return 'warning';
    return 'danger';
  }

  onFileSelected(event: Event): void {
    const file = (event.target as HTMLInputElement)?.files?.[0];
    if (file) {
      this.selectedFile = file;
      this.onUpload();
    }
  }

  onUpload(): void {
    if (this.selectedFile) {
      this.isUploading.set(true);
      this.fileUploadService.uploadFile(this.selectedFile, this.localStorageService.getItem('uuid')).subscribe(
        (progress: number) => this.uploadProgress.update((progress) => progress),
        (error) => {
          console.error('Upload failed:', error);
          this.isUploading.set(false);
        },
        () => {
          console.log('Upload successful');
          this.isUploading.set(false);
        }
      );
    }
  }

  private unsubscribeFromChannels(): void {
    if (this.pathogenChannelSubscription) {
      this.pathogenChannelSubscription.unsubscribe();
      this.actionCableService.unsubscribeFromChannel('PathogenDetectionChannel')
    }
    if (this.pathogenEnrichmentSubscription) {
      this.pathogenEnrichmentSubscription.unsubscribe();
      this.actionCableService.unsubscribeFromChannel('PathogenEnrichmentChannel')
    }
    console.log('Unsubscribed from all channels.');
  }
  openEnrichmentModal(detection: Detection): void {
    try {
      detection.processed_image = this.img;
      this.enrichmentModal.open(detection);
    } catch {
     // this.isCorruptedEnrichmentVisible.update(() => true);
    }
  }

  trackByDetectionId(index: number, detection: Detection): string {
    return detection.detection_id;
  }

  closeEnrichmentCorruption() {
    this.isCorruptedEnrichmentVisible.set(false);
  }

  handleEnrichmentCorruptionChange(event: any) {
    this.isCorruptedEnrichmentVisible.update(() => event);
  }
}
