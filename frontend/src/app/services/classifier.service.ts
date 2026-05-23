import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';
import { Observable } from 'rxjs';

export interface ClassifierPrediction {
  query: string;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
}

@Injectable({
  providedIn: 'root',
})
export class ClassifierService {
  constructor(private http: HttpClient) {}

  predict(query: string): Observable<ClassifierPrediction> {
    return this.http.post<ClassifierPrediction>(`${environment.classifierUrl}/predict`, { query });
  }
}
