import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../environments/environment';

@Injectable({
  providedIn: 'root',
})
export class ApiService {
  constructor(private http: HttpClient) {}

  post<T>(path: string, body: any) {
    return this.http.post<T>(`${environment.apiBaseUrl}${path}`, body);
  }

  get<T>(path: string) {
    return this.http.get<T>(`${environment.apiBaseUrl}${path}`);
  }
}
