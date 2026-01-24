export const API_BASE = '/api'; // Relative path if proxied, or fully qualified if CORS allowed

export interface Metric {
  model_name: string;
  rmse: number;
  mae: number;
  mape: number;
  trained_at: string;
}

export interface Status {
  has_data: boolean;
  models_trained: boolean;
}

export interface ImageCollection {
  images: Record<string, any>;
}

export interface ForecastResponse {
  image: any;
}

export async function fetchStatus(): Promise<Status> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error('Failed to fetch status');
  return res.json();
}

export async function fetchMetrics(): Promise<{ items: Metric[] }> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch metrics');
  return res.json();
}

export async function fetchEDA(): Promise<ImageCollection> {
  const res = await fetch(`${API_BASE}/eda`);
  if (!res.ok) throw new Error('Failed to fetch EDA');
  return res.json();
}

export async function fetchFeatureImportance(): Promise<ImageCollection> {
  const res = await fetch(`${API_BASE}/feature-importance?model=lgbm`);
  if (!res.ok) throw new Error('Failed to fetch Feature Importance');
  return res.json();
}

export async function fetchHoldout(): Promise<{ image: any; start: string; end: string }> {
  const res = await fetch(`${API_BASE}/holdout`);
  if (!res.ok) throw new Error('Failed to fetch Holdout');
  return res.json();
}

export async function fetchForecast(horizon: number): Promise<ForecastResponse> {
  const res = await fetch(`${API_BASE}/forecast_plot?horizon=${horizon}`);
  if (!res.ok) throw new Error('Failed to fetch Forecast');
  return res.json();
}

export async function trainModels(): Promise<void> {
  const res = await fetch(`${API_BASE}/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  if (!res.ok) throw new Error('Training failed');
}

export async function uploadData(file: File): Promise<void> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
}
