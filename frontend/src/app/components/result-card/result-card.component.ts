import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { QueryResponse } from '../../models/query.model';

@Component({
  selector: 'app-result-card',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './result-card.component.html',
  styleUrls: ['./result-card.component.scss'],
})
export class ResultCardComponent {
  @Input() result: QueryResponse | null = null;
}
