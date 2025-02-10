import {
  Component,
  Inject,
} from '@angular/core';
import {DecimalPipe, DOCUMENT, NgClass, NgForOf, NgIf} from '@angular/common';
import {CardBodyComponent, CardComponent, ColComponent, RowComponent, TableDirective} from "@coreui/angular";
import { faCircleDot,faCoffee } from '@fortawesome/free-solid-svg-icons';
import {FaIconComponent} from "@fortawesome/angular-fontawesome";

@Component({
  standalone: true,
  templateUrl: 'model.component.html',
  styleUrls: ['model.component.css'],
  imports: [
    DecimalPipe,
    NgIf,
    NgForOf,
    NgClass,
    CardBodyComponent,
    CardComponent,
    ColComponent,
    RowComponent,
    TableDirective,
    FaIconComponent
  ]
})
export class ModelComponent {
  faCoffee = faCoffee
  faCircleDot = faCircleDot

 public layers  = [
    {
      name: 'input_layer_2',
      type: 'InputLayer',
      outputShape: '(None, 300, 300, 3)',
      params: 0,
      description: 'Entry point for 300x300 RGB plant disease images. Receives raw pixel data and normalizes using EfficientNetV2 preprocessing.',
      isExpanded: false
    },
    {
      name: 'efficientnetv2-b2',
      type: 'Functional',
      outputShape: '(None, 10, 10, 1408)',
      params: 8769374,
      description: 'Pretrained ImageNet backbone frozen during initial training. Extracts spatial features at 1/30th resolution using mobile inverted bottleneck convolutions. Provides foundational patterns for disease recognition.',
      isExpanded: false
    },
    {
      name: 'separable_block',
      type: 'SeparableBlock',
      outputShape: '(None, 10, 10, 512)',
      params: 741248,
      description: 'Custom depthwise separable convolution block. Combines depthwise spatial filtering with pointwise channel mixing. Uses SWISH activation (x * sigmoid(x)) for smooth gradient flow and batch normalization for stable training.',
      isExpanded: false
    },
    {
      name: 'hybrid_attention_block',
      type: 'HybridAttentionBlock',
      outputShape: '(None, 10, 10, 512)',
      params: 58401,
      description: 'Dual attention mechanism for plant pathology: Channel Attention (squeeze-excitation) emphasizes disease-specific feature maps, Spatial Attention (7x7 conv) highlights lesion locations. Critical for ignoring healthy tissue regions.',
      isExpanded: false
    },
    {
      name: 'global_average_pooling2d_1',
      type: 'GlobalAveragePooling2D',
      outputShape: '(None, 512)',
      params: 0,
      description: 'Reduces 10x10 feature maps to 512D vector by spatial averaging. Maintains channel-wise disease pattern information while removing spatial redundancy.',
      isExpanded: false
    },
    {
      name: 'dense_2',
      type: 'Dense',
      outputShape: '(None, 512)',
      params: 262656,
      description: 'Fully-connected layer with SWISH activation. Learns non-linear combinations of disease biomarkers. Regularized with L2 weight decay (λ=1e-4) to prevent overfitting on rare pathogens.',
      isExpanded: false
    },
    {
      name: 'batch_normalization_2',
      type: 'BatchNormalization',
      outputShape: '(None, 512)',
      params: 2048,
      description: 'Standardizes layer inputs to μ=0, σ=1. Enables higher learning rates and reduces dependency on careful initialization. Critical given class imbalance in plant diseases.',
      isExpanded: false
    },
    {
      name: 'dropout',
      type: 'Dropout',
      outputShape: '(None, 512)',
      params: 0,
      description: 'Regularization layer that randomly zeros 30% of activations during training. Prevents co-adaptation of disease features, forcing robust biomarker learning.',
      isExpanded: false
    },
    {
      name: 'dense_4',
      type: 'Dense',
      outputShape: '(None, 146)',
      params: 74898,
      description: 'Final classification layer producing logits for 146 plant disease classes. Weights represent pathogen-specific decision boundaries in high-dimensional feature space.',
      isExpanded: false
    },
    {
      name: 'temperature_scaling',
      type: 'TemperatureScaling',
      outputShape: '(None, 146)',
      params: 1,
      description: 'Post-hoc calibration layer learned on validation set. Scales logits by T=1.0 (initially) to better align confidence with accuracy. Critical for reliable disease prognosis.',
      isExpanded: false
    }
  ];
  constructor(
    @Inject(DOCUMENT) private document: Document
  ) {
  }

  toggleDescription(index: number) {
    this.layers[index].isExpanded = !this.layers[index].isExpanded;
    this.layers.forEach((layer, i) => {
      if(i !== index) layer.isExpanded = false;
    });
  }
  // getLayerIcon(layerType: string): string {
  //   const iconMap: {[key: string]: string} = {
  //     'InputLayer': 'faCircleDot',
  //     'Functional': 'fa-solid fa-brain',
  //     'SeparableBlock': 'fa-solid fa-filter',
  //     'HybridAttentionBlock': 'fa-solid fa-bullseye',
  //     'GlobalAveragePooling2D': 'fa-solid fa-compress',
  //     'Dense': 'fa-solid fa-project-diagram',
  //     'BatchNormalization': 'fa-solid fa-sliders',
  //     'Dropout': 'fa-solid fa-shield',
  //     'TemperatureScaling': 'fa-solid fa-thermometer'
  //   };
  //   return iconMap[layerType] || 'fa-solid fa-layer-group';
  // }
}

