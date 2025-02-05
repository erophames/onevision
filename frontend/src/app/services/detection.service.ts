import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpEventType, HttpResponse, HttpEvent } from '@angular/common/http';
import {Observable} from "rxjs";
import {catchError} from "rxjs/operators";

@Injectable({
  providedIn: 'root'
})
export class DetectionService {
  private apiUrl = 'https://api.rednode.co.za/detections';

  constructor(private http: HttpClient) {}

  topDiseases(): Observable<any[]>{
    return this.http.get<any[]>(this.apiUrl + '/dashboard', {
      headers: new HttpHeaders()
    }).pipe(
      catchError(error => {
        console.error('Error fetching dashboard:', error);
        throw error;
      })
    );
  }
  getDetections(): Observable<any[]> {
    return this.http.get<any[]>(this.apiUrl, {
      headers: new HttpHeaders()
    }).pipe(
      catchError(error => {
        console.error('Error fetching detections:', error);
        throw error;
      })
    );
  }
}

