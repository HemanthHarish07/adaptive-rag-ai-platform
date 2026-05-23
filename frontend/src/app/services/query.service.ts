import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable, tap } from 'rxjs';
import { ApiService } from './api.service';
import { QueryHistoryEntry, QueryResponse } from '../models/query.model';

@Injectable({
  providedIn: 'root',
})
export class QueryService {
  private readonly historyKey = 'ai-platform-history';
  private historySubject = new BehaviorSubject<QueryHistoryEntry[]>(this.loadHistory());
  history$ = this.historySubject.asObservable();

  constructor(private api: ApiService) {}

  submitQuery(question: string): Observable<QueryResponse> {
    return this.api.post<QueryResponse>('/query', { question }).pipe(
      tap((response) => this.saveQuery(response))
    );
  }

  private loadHistory(): QueryHistoryEntry[] {
    const payload = localStorage.getItem(this.historyKey);
    if (!payload) {
      return [];
    }

    try {
      return JSON.parse(payload) as QueryHistoryEntry[];
    } catch {
      return [];
    }
  }

  private saveQuery(response: QueryResponse): void {
    const entry: QueryHistoryEntry = {
      ...response,
      timestamp: new Date().toISOString(),
    };
    const history = [entry, ...this.historySubject.value].slice(0, 20);
    this.historySubject.next(history);
    localStorage.setItem(this.historyKey, JSON.stringify(history));
  }

  loadHistoryEntry(entry: QueryHistoryEntry): QueryHistoryEntry {
    return entry;
  }
}
