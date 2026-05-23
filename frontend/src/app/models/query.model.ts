export interface QueryResponse {
  question: string;
  answer: string;
  sources: string[];
  processing_steps: string[];
  predicted_intent?: string;
  intent_confidence?: number;
  intent_probabilities?: Record<string, number> | null;
  selected_graph_route?: string;
  retrieval_strategy?: string;
  classifier_request_trace?: string[] | null;
}

export interface QueryHistoryEntry extends QueryResponse {
  timestamp: string;
}
