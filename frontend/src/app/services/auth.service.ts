import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { environment } from '../../environments/environment';
import { LoginResponse, RegisterRequest } from '../models/auth.model';

@Injectable({
  providedIn: 'root',
})
export class AuthService {
  private readonly tokenKey = 'ai-platform-token';
  private readonly userKey = 'ai-platform-user';
  private userSubject = new BehaviorSubject<string | null>(this.getStoredUsername());
  user$ = this.userSubject.asObservable();

  constructor(private http: HttpClient) {}

  login(username: string, password: string): Observable<LoginResponse> {
    const body = new URLSearchParams();
    body.set('username', username);
    body.set('password', password);

    return this.http.post<LoginResponse>(`${environment.apiBaseUrl}/token`, body.toString(), {
      headers: new HttpHeaders({ 'Content-Type': 'application/x-www-form-urlencoded' }),
    }).pipe(
      tap((response) => {
        localStorage.setItem(this.tokenKey, response.access_token);
        localStorage.setItem(this.userKey, username);
        this.userSubject.next(username);
      })
    );
  }

  register(request: RegisterRequest) {
    return this.http.post(`${environment.apiBaseUrl}/register`, request);
  }

  logout() {
    localStorage.removeItem(this.tokenKey);
    localStorage.removeItem(this.userKey);
    this.userSubject.next(null);
  }

  getToken(): string | null {
    return localStorage.getItem(this.tokenKey);
  }

  isLoggedIn(): boolean {
    return !!this.getToken();
  }

  private getStoredUsername(): string | null {
    return localStorage.getItem(this.userKey);
  }
}
