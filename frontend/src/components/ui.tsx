import React from 'react';
import Plot from 'react-plotly.js';

export const Card: React.FC<{ title: string; children: React.ReactNode; className?: string }> = ({ title, children, className }) => (
    <div className={`card ${className || ''}`}>
        <h2>{title}</h2>
        {children}
    </div>
);

export const Button: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'outline' }> = ({ variant = 'primary', className, ...props }) => (
    <button className={`btn btn-${variant} ${className || ''}`} {...props} />
);

export const Select: React.FC<React.SelectHTMLAttributes<HTMLSelectElement>> = (props) => (
    <select {...props} />
);

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = (props) => (
    <input {...props} />
);

export const LoadingSpinner = () => (
    <div className="animate-spin" style={{ width: '24px', height: '24px', borderRadius: '50%', border: '2px solid transparent', borderTopColor: 'var(--color-primary)' }} />
);

export const PlotDisplay: React.FC<{ data: any; loading?: boolean }> = ({ data, loading }) => {
    if (loading) return <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><LoadingSpinner /></div>;
    if (!data) return <div style={{ height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--color-text-muted)' }}>Select an item to view</div>;
    return (
        <div className="hero" style={{ minHeight: '500px' }}>
            <Plot
                data={data.data}
                layout={{ ...data.layout, autosize: true }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ responsive: true, displayModeBar: false }}
            />
        </div>
    );
};

export const Modal: React.FC<{ isOpen: boolean; onClose: () => void; title: string; children: React.ReactNode }> = ({ isOpen, onClose, title, children }) => {
    if (!isOpen) return null;
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <header className="header" style={{ marginBottom: '1rem', paddingBottom: '0.5rem' }}>
                    <h3>{title}</h3>
                    <button className="btn btn-outline" style={{ padding: '0.25rem 0.5rem' }} onClick={onClose}>✕</button>
                </header>
                <div className="body">
                    {children}
                </div>
            </div>
        </div>
    );
};
