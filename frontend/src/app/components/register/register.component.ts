import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { AuthService } from '../../services/auth.service';
import { RegisterRequest } from '../../models/auth.model';

@Component({
  selector: 'app-register',
  standalone: true,
  imports: [CommonModule, ReactiveFormsModule, RouterLink],
  templateUrl: './register.component.html',
  styleUrls: ['./register.component.scss'],
})
export class RegisterComponent {
  private fb = inject(FormBuilder);
  private auth = inject(AuthService);
  private router = inject(Router);

  form = this.fb.group({
    username: ['', [Validators.required, Validators.minLength(3)]],
    email: ['', [Validators.required, Validators.email]],
    password: ['', [Validators.required, Validators.minLength(6)]],
  });

  loading = false;
  errorMessage = '';
  successMessage = '';

  submit() {
    if (this.form.invalid) {
      this.errorMessage = 'Please complete all fields with valid values.';
      return;
    }

    this.loading = true;
    this.errorMessage = '';
    this.successMessage = '';

    const request = this.form.value as RegisterRequest;
    this.auth.register(request).subscribe({
      next: () => {
        this.loading = false;
        this.successMessage = 'Account created successfully. Redirecting to login...';
        setTimeout(() => this.router.navigate(['/login']), 1200);
      },
      error: (error) => {
        this.loading = false;
        this.errorMessage = error?.error?.detail || 'Registration failed. Username may already exist.';
      },
    });
  }
}
