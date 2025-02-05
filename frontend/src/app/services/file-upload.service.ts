import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpEventType, HttpResponse, HttpEvent } from '@angular/common/http';
import { Observable, Subject } from 'rxjs';
import { catchError, map } from 'rxjs/operators';

@Injectable({
  providedIn: 'root'
})
export class FileUploadService {
  private apiUrl = 'https://api.rednode.co.za/detections';

  constructor(private http: HttpClient) {}

  uploadFile(file: File, userId: string): Observable<any> {
    const formData: FormData = new FormData();
    formData.append('image', file, file.name);
    formData.append('user_id',userId);

    const progressSubject = new Subject<any>();

    this.http.post(this.apiUrl, formData, {
      headers: new HttpHeaders(),
      observe: 'events',
      reportProgress: true
    }).pipe(
      map((event: HttpEvent<any>) => {
        switch (event.type) {
          case HttpEventType.UploadProgress:
            if (event.total) {
              const progress = Math.round(100 * event.loaded / event.total);
              progressSubject.next(progress);
            }
            break;
          case HttpEventType.Response:
            console.log('File uploaded successfully', event.body);
            progressSubject.complete();
            return event.body;
        }
      }),
      catchError(error => {
        console.error('Error uploading file:', error);
        progressSubject.error(error);
        return [];
      })
    ).subscribe();

    return progressSubject.asObservable();
  }
}

