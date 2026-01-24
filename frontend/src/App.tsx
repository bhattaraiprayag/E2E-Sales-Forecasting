import { useState, useEffect, useCallback } from 'react';
import { Card, Button, Select, Input, PlotDisplay, Modal, LoadingSpinner } from './components/ui';
import {
  fetchStatus, fetchMetrics, fetchEDA, fetchFeatureImportance, fetchHoldout, fetchForecast,
  trainModels, uploadData, type Status, type Metric
} from './services/api';
import './index.css';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'eda', label: 'EDA' },
  { id: 'insights', label: 'Model Insights' },
  { id: 'backtesting', label: 'Backtesting' },
  { id: 'forecast', label: 'Forecasts' },
];

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [status, setStatus] = useState<Status>({ has_data: false, models_trained: false });
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<Metric[]>([]);

  // Data for tabs (Plots are now JSON objects, not URLs)
  const [edaPlots, setEdaPlots] = useState<Record<string, any>>({});
  const [selectedEda, setSelectedEda] = useState<string>('');

  const [fiPlots, setFiPlots] = useState<Record<string, any>>({});
  const [selectedFi, setSelectedFi] = useState<string>('');

  const [holdoutPlot, setHoldoutPlot] = useState<any>(null);
  const [forecastPlot, setForecastPlot] = useState<any>(null);
  const [horizon, setHorizon] = useState(30);

  // Modal State
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const loadData = useCallback(async () => {
    try {
      const s = await fetchStatus();
      setStatus(s);

      if (s.models_trained) {
        const m = await fetchMetrics();
        setMetrics(m.items);
      }
    } catch (e) {
      console.error(e);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Lazy load tab data
  useEffect(() => {
    if (activeTab === 'eda' && status.has_data && !Object.keys(edaPlots).length) {
      setLoading(true);
      fetchEDA().then(res => {
        setEdaPlots(res.images);
        // Default to historical
        setSelectedEda('historical');
      }).finally(() => setLoading(false));
    }
    if (activeTab === 'insights' && status.models_trained && !Object.keys(fiPlots).length) {
      setLoading(true);
      fetchFeatureImportance().then(res => {
        setFiPlots(res.images);
        // Default to lgbm_gain
        setSelectedFi('lgbm_gain');
      }).finally(() => setLoading(false));
    }
    if (activeTab === 'backtesting' && status.models_trained && !holdoutPlot) {
      setLoading(true);
      fetchHoldout().then(res => setHoldoutPlot(res.image)).finally(() => setLoading(false));
    }
    if (activeTab === 'forecast' && status.models_trained && !forecastPlot) {
      setLoading(true);
      fetchForecast(horizon).then(res => setForecastPlot(res.image)).finally(() => setLoading(false));
    }
  }, [activeTab, status, edaPlots, fiPlots, holdoutPlot, forecastPlot, horizon]);

  const handleTrain = async () => {
    setLoading(true);
    try {
      await trainModels();
      alert('Training started/completed!');
      await loadData();
      // Clear cache
      setFiPlots({});
      setHoldoutPlot(null);
      setForecastPlot(null);
    } catch (e: any) {
      alert('Training failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async () => {
    if (!uploadFile) return;
    setLoading(true);
    try {
      await uploadData(uploadFile);
      setIsUploadOpen(false);
      alert('Upload successful');
      await loadData();
    } catch (e: any) {
      alert('Upload failed: ' + e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleForecastGen = async () => {
    setLoading(true);
    try {
      const res = await fetchForecast(horizon);
      setForecastPlot(res.image);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="header">
        <h1>Sales Forecasting</h1>
        <div className="actions">
          <Button variant="outline" onClick={() => setIsUploadOpen(true)}>Upload Data</Button>
          <Button onClick={handleTrain} disabled={loading}>{loading ? <LoadingSpinner /> : 'Train Models'}</Button>
        </div>
      </div>

      <nav className="tabs">
        {TABS.map(tab => {
          const disabled = (tab.id === 'eda' && !status.has_data) ||
            (['insights', 'backtesting', 'forecast'].includes(tab.id) && !status.models_trained);
          return (
            <div
              key={tab.id}
              className={`tab ${activeTab === tab.id ? 'active' : ''} ${disabled ? 'disabled' : ''}`}
              onClick={() => !disabled && setActiveTab(tab.id)}
            >
              {tab.label}
            </div>
          );
        })}
      </nav>

      <div className="content">
        {activeTab === 'overview' && (
          <Card title="Model Performance (Holdout)">
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>MAPE</th>
                    <th>Trained At</th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.length === 0 ? (
                    <tr><td colSpan={5} style={{ textAlign: 'center', padding: '2rem' }}>No models trained yet.</td></tr>
                  ) : (
                    metrics.map(m => (
                      <tr key={m.model_name}>
                        <td>{m.model_name.toUpperCase()}</td>
                        <td>{m.rmse?.toFixed(2)}</td>
                        <td>{m.mae?.toFixed(2)}</td>
                        <td>{m.mape?.toFixed(2)}%</td>
                        <td>{new Date(m.trained_at).toLocaleString()}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        )}

        {activeTab === 'eda' && (
          <Card title="Exploratory Data Analysis">
            <div className="row" style={{ marginBottom: '1rem' }}>
              <label>Select plot:</label>
              <Select value={selectedEda} onChange={(e) => setSelectedEda(e.target.value)} style={{ flex: 1 }}>
                {Object.keys(edaPlots).map((key) => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </Select>
            </div>
            {/* edaPlots[selectedEda] gives the plot data object */}
            <PlotDisplay data={edaPlots[selectedEda]} loading={loading} />
          </Card>
        )}

        {activeTab === 'insights' && (
          <Card title="Feature Importance / SHAP">
            <div className="row" style={{ marginBottom: '1rem' }}>
              <label>Select insight:</label>
              <Select value={selectedFi} onChange={(e) => setSelectedFi(e.target.value)} style={{ flex: 1 }}>
                {Object.keys(fiPlots).map((key) => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </Select>
            </div>
            <PlotDisplay data={fiPlots[selectedFi]} loading={loading} />
          </Card>
        )}

        {activeTab === 'backtesting' && (
          <Card title="Holdout Performance">
            <PlotDisplay data={holdoutPlot} loading={loading} />
          </Card>
        )}

        {activeTab === 'forecast' && (
          <Card title="Future Forecasts">
            <div className="row" style={{ marginBottom: '1rem', flexWrap: 'wrap' }}>
              <label>Horizon (days):</label>
              <Input type="number" min={1} max={180} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
              <Button onClick={handleForecastGen} disabled={loading}>Generate</Button>
            </div>
            <PlotDisplay data={forecastPlot} loading={loading} />
          </Card>
        )}
      </div>

      <Modal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} title="Upload Data">
        <p style={{ marginBottom: '1rem', color: 'var(--color-text-muted)' }}>
          Upload a CSV or XLSX file with columns: <code>date, daily_sales, marketing_spend, is_holiday</code>.
        </p>
        <Input type="file" accept=".csv,.xlsx" onChange={(e) => setUploadFile(e.target.files?.[0] || null)} style={{ width: '100%', marginBottom: '1rem' }} />
        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
          <Button onClick={handleUpload} disabled={!uploadFile || loading}>
            {loading ? <LoadingSpinner /> : 'Upload'}
          </Button>
        </div>
      </Modal>
    </div>
  );
}

export default App;
