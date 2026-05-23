import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-route-badge',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './route-badge.component.html',
  styleUrls: ['./route-badge.component.scss'],
})
export class RouteBadgeComponent {
  @Input() label = '';
  @Input() value = '';
  @Input() variant: 'danger' | 'warning' | 'success' | 'info' = 'info';
}
