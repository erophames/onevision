import {Injectable, signal} from '@angular/core';
import {ChartData, ChartOptions, ChartType, ScaleOptions} from 'chart.js';
import {DetectionService} from "../../services/detection.service";
import {getStyle} from '@coreui/utils';

export interface IChartProps {
  data?: ChartData;
  labels?: any;
  options?: ChartOptions;
  colors?: any;
  type: ChartType;
  legend?: any;

  [propName: string]: any;
}

@Injectable({
  providedIn: 'any'
})
export class DashboardChartsData {
  public topDiseases = signal<any>(null);

  constructor(private detectionService: DetectionService) {
  }

  public mainChart: { type: ChartType; data?: ChartData; options?: ChartOptions } = {type: 'bar'};

  // Fetch and update chart data for top diseases
  fetchTopDiseases() {
    this.detectionService.topDiseases().subscribe((data) => {

      this.topDiseases.set(data);
      const labels = Object.keys(data);
      const values = Object.values(data);

      this.mainChart.data = {
        labels,
        datasets: [
          {
            label: 'Top Diseases',
            data: values,
            backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
          }
        ]
      };

      this.mainChart.options = {
        responsive: true,
        scales: this.getScales() // Use the getScales function here
      };
    });
  }

  // Initialize the chart with default or custom settings
  initChart(data?: { labels: string[]; values: number[] }) {
    const labels = data ? data.labels : [];
    const values = data ? data.values : [];

    this.mainChart.data = {
      labels,
      datasets: [
        {
          label: 'Top Diseases',
          data: values,
          backgroundColor: ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF']
        }
      ]
    };

    this.mainChart.options = {
      responsive: true,
      scales: this.getScales() // Apply scales settings
    };
  }

  // Return chart scales configuration
  getScales() {
    const colorBorderTranslucent = getStyle('--cui-border-color-translucent');
    const colorBody = getStyle('--cui-body-color');

    const scales: ScaleOptions<any> = {
      x: {
        grid: {
          color: colorBorderTranslucent,
          drawOnChartArea: false
        },
        ticks: {
          color: colorBody
        }
      },
      y: {
        border: {
          color: colorBorderTranslucent
        },
        grid: {
          color: colorBorderTranslucent
        },
        max: 10, // You can change the max as per your data range
        beginAtZero: true,
        ticks: {
          color: colorBody,
          maxTicksLimit: 5,
          stepSize: Math.ceil(250 / 5)
        }
      }
    };
    return scales;
  }
}
