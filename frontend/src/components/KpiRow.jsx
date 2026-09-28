import React from 'react';
import './KpiRow.css';

export default function KpiRow({ summary, framesProcessed }) {
  const total = summary?.total || 0;
  const helmet = summary?.NO_HELMET || 0;
  const triple = summary?.TRIPLE_RIDING || 0;
  const seatbelt = summary?.NO_SEATBELT || 0;
  const signal = summary?.SIGNAL_JUMP || 0;
  const wrongWay = summary?.WRONG_WAY || 0;

  return (
    <div className="kpi-container">
      <div className="kpi-strip">
        <div className="kpi-block">
          <div className="kpi-value mono">{total}</div>
          <div className="kpi-label govt-badge">Total Violations</div>
        </div>
        
        <div className="kpi-block">
          <div className="kpi-value mono">{helmet}</div>
          <div className="kpi-label govt-badge">No Helmet</div>
        </div>
        
        <div className="kpi-block">
          <div className="kpi-value mono">{triple}</div>
          <div className="kpi-label govt-badge">Triple Riding</div>
        </div>

        <div className="kpi-block">
          <div className="kpi-value mono">{seatbelt}</div>
          <div className="kpi-label govt-badge">No Seatbelt</div>
        </div>

        <div className="kpi-block">
          <div className="kpi-value mono">{signal}</div>
          <div className="kpi-label govt-badge">Signal Jump</div>
        </div>

        <div className="kpi-block">
          <div className="kpi-value mono">{wrongWay}</div>
          <div className="kpi-label govt-badge">Wrong Way</div>
        </div>

        <div className="kpi-block">
          <div className="kpi-value mono">{framesProcessed || 0}</div>
          <div className="kpi-label govt-badge">Frames Processed</div>
        </div>
      </div>
    </div>
  );
}
