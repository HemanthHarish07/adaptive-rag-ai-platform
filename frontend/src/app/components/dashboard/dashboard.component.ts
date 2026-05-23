import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router } from '@angular/router';
import { NavbarComponent } from '../navbar/navbar.component';
import { RouteBadgeComponent } from '../route-badge/route-badge.component';
import { ResultCardComponent } from '../result-card/result-card.component';
import { QueryService } from '../../services/query.service';
import { AuthService } from '../../services/auth.service';
import { ClassifierService, ClassifierPrediction } from '../../services/classifier.service';
import { QueryHistoryEntry, QueryResponse } from '../../models/query.model';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, NavbarComponent, RouteBadgeComponent, ResultCardComponent],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent {
  private fb = inject(FormBuilder);
  private queryService = inject(QueryService);
  private classifierService = inject(ClassifierService);
  private auth = inject(AuthService);
  private router = inject(Router);

  form = this.fb.group({
    question: ['', [Validators.required, Validators.minLength(5)]],
  });

  loading = false;
  classifierLoading = false;
  errorMessage = '';
  currentResult: QueryResponse | null = null;
  classifierPreview: ClassifierPrediction | null = null;
  selectedHistory: QueryHistoryEntry | null = null;
  user$ = this.auth.user$;
  history$ = this.queryService.history$;

  constructor() {
    if (!this.auth.isLoggedIn()) {
      this.router.navigate(['/login']);
    }
  }

  submit() {
    if (this.form.invalid) {
      this.errorMessage = 'Type a learning question to continue.';
      return;
    }

    this.loading = true;
    this.errorMessage = '';
    this.classifierPreview = null;
    const question = this.form.value.question!;

    this.queryService.submitQuery(question).subscribe({
      next: (result) => {
        this.currentResult = result;
        this.selectedHistory = null;
        this.loading = false;
      },
      error: (error) => {
        this.loading = false;
        this.errorMessage = error?.error?.detail || 'Failed to process the query. Please try again.';
      },
    });
  }

  showHistory(entry: QueryHistoryEntry) {
    this.currentResult = entry;
    this.selectedHistory = entry;
    this.errorMessage = '';
    this.classifierPreview = null;
  }

  formatConfidence(value: number | null | undefined): string {
    return value != null ? `${(value * 100).toFixed(1)}%` : 'n/a';
  }


  formatClassifierTrace(step: unknown): string {
    // Step can be a string or an object; render deterministically.
    if (step == null) return '';

    if (typeof step === 'string') return step;

    if (typeof step === 'object') {
      const s = step as Record<string, unknown>;
      const phase = (s['phase'] ?? s['request_phase'] ?? s['type'] ?? s['label'] ?? 'request') as unknown;
      const status = (s['status_code'] ?? s['status'] ?? s['code'] ?? null) as unknown;
      const method = (s['method'] ?? null) as unknown;
      const url = (s['path'] ?? s['url'] ?? null) as unknown;

      // If this looks like a request entry
      if (status == null) {
        const m = typeof method === 'string' ? method : 'POST';
        const u = typeof url === 'string' ? url : '/predict';
        return `→ request: ${m} ${u}`;
      }

      // Response entry
      if (typeof status === 'number') {
        const m = typeof method === 'string' ? method : 'POST';
        const u = typeof url === 'string' ? url : '/predict';
        return `→ response: ${status} OK`;
      }

      // Best-effort fallback
      const phaseStr = typeof phase === 'string' ? phase : 'request';
      const statusStr = typeof status === 'string' ? status : String(status);
      return `→ ${phaseStr}: ${statusStr}`;
    }

    return String(step);
  }

  getBadgeVariant(text: string | undefined): 'danger' | 'warning' | 'success' | 'info' {

    if (!text) {
      return 'info';
    }
    const value = text.toLowerCase();
    if (value.includes('web')) {
      return 'danger';
    }
    if (value.includes('vector') || value.includes('retrieve')) {
      return 'success';
    }
    if (value.includes('debug') || value.includes('coding')) {
      return 'warning';
    }
    return 'info';
  }

  truncateHistoryQuestion(q: string | undefined | null): string {
    const text = q ?? '';
    if (text.length <= 40) return text;
    return text.slice(0, 40) + '...';
  }

  probeClassifier() {

    const query = this.form.value.question;
    if (!query || query.length < 4) {
      this.errorMessage = 'Enter a longer query to probe the classifier.';
      return;
    }
    this.classifierLoading = true;
    this.classifierPreview = null;
    this.classifierService.predict(query).subscribe({
      next: (prediction) => {
        this.classifierPreview = prediction;
        this.classifierLoading = false;
      },
      error: (error) => {
        this.classifierLoading = false;
        this.errorMessage = error?.error?.detail || 'Classifier probe failed.';
      },
    });
  }
}
