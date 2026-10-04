import React from 'react';
import { useSettings } from '../contexts/SettingsContext';
import './Settings.css';

export default function Settings() {
  const { settings, setSettings, handleClearLog } = useSettings();
  const [cleared, setCleared] = React.useState(false);

  const handleChange = (key, value) => {
    setSettings({ ...settings, [key]: value });
  };

  return (
    <div className="page-container" style={{ maxWidth: '600px', width: '100%', margin: '0 auto' }}>
      <h2 className="page-title govt-badge">Configuration & Rules</h2>
      
      <div className="settings-panel">
        <div className="section-header">
          <span>Model Config</span>
        </div>
        
        <div className="control-group">
          <div className="setting-header">
            <label>Confidence Threshold</label>
            <span className="setting-value mono">{settings.confThreshold.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            className="native-slider"
            min="0.1" max="0.9" step="0.05"
            value={settings.confThreshold}
            onChange={(e) => handleChange('confThreshold', parseFloat(e.target.value))}
          />
        </div>

        <div className="control-group">
          <div className="setting-header">
            <label>Overlap Threshold</label>
            <span className="setting-value mono">{settings.overlapThreshold.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            className="native-slider"
            min="0.1" max="0.8" step="0.05"
            value={settings.overlapThreshold}
            onChange={(e) => handleChange('overlapThreshold', parseFloat(e.target.value))}
          />
        </div>
      </div>

      <div className="settings-panel">
        <div className="section-header">
          <span>Violation Rules</span>
        </div>
        
        <label className="native-checkbox-label">
          <input 
            type="checkbox" 
            className="native-checkbox"
            checked={settings.checkHelmet}
            onChange={(e) => handleChange('checkHelmet', e.target.checked)}
          />
          <span className="label-text">No Helmet Detection</span>
        </label>

        <label className="native-checkbox-label">
          <input 
            type="checkbox" 
            className="native-checkbox"
            checked={settings.checkTriple}
            onChange={(e) => handleChange('checkTriple', e.target.checked)}
          />
          <span className="label-text">Triple Riding Detection</span>
        </label>

        {settings.checkTriple && (
          <div className="control-group child-control">
            <label className="sub-label">Triple Riding Threshold</label>
            <div className="number-input-wrapper">
              <input 
                type="number" 
                className="native-number"
                min="2" max="5" 
                value={settings.tripleThreshold}
                onChange={(e) => handleChange('tripleThreshold', parseInt(e.target.value, 10))}
              />
              <span className="number-suffix">riders</span>
            </div>
          </div>
        )}

        <label className="native-checkbox-label">
          <input
            type="checkbox"
            className="native-checkbox"
            checked={settings.checkSeatbelt}
            onChange={(e) => handleChange('checkSeatbelt', e.target.checked)}
          />
          <span className="label-text">No Seatbelt Detection</span>
        </label>

        <label className="native-checkbox-label">
          <input
            type="checkbox"
            className="native-checkbox"
            checked={settings.checkSignal}
            onChange={(e) => handleChange('checkSignal', e.target.checked)}
          />
          <span className="label-text">Signal Jump Detection</span>
        </label>

        {settings.checkSignal && (
          <div className="control-group child-control">
            <div className="setting-header">
              <label className="sub-label">Stop line position (from top of video)</label>
              <span className="setting-value mono">{settings.stopLineY}%</span>
            </div>
            <input
              type="range"
              className="native-slider"
              min="10" max="95" step="1"
              value={settings.stopLineY}
              onChange={(e) => handleChange('stopLineY', parseInt(e.target.value, 10))}
            />
          </div>
        )}

        <label className="native-checkbox-label" style={{ marginBottom: 0 }}>
          <input
            type="checkbox"
            className="native-checkbox"
            checked={settings.checkWrongWay}
            onChange={(e) => handleChange('checkWrongWay', e.target.checked)}
          />
          <span className="label-text">Wrong Way / Lane Violation Detection</span>
        </label>

        {settings.checkWrongWay && (
          <div className="control-group child-control">
            <label className="sub-label">Normal traffic direction in the video</label>
            <select
              className="native-number"
              style={{ width: '100%' }}
              value={settings.expectedDirection}
              onChange={(e) => handleChange('expectedDirection', e.target.value)}
            >
              <option value="down">Towards camera (moving down)</option>
              <option value="up">Away from camera (moving up)</option>
              <option value="left">Moving left</option>
              <option value="right">Moving right</option>
            </select>
          </div>
        )}
      </div>

      <div className="settings-panel">
        <div className="section-header">
          <span>Input Mode</span>
        </div>
        
        <div className="native-segmented-control">
          <button 
            className={settings.inputMode === 'image' ? 'active' : ''}
            onClick={() => handleChange('inputMode', 'image')}
          >
            Image
          </button>
          <button 
            className={settings.inputMode === 'video' ? 'active' : ''}
            onClick={() => handleChange('inputMode', 'video')}
          >
            Video
          </button>
        </div>
      </div>

      <div style={{ marginTop: '32px' }}>
        <button 
          className="btn-destructive w-full" 
          onClick={async () => {
            await handleClearLog();
            setCleared(true);
            setTimeout(() => setCleared(false), 2000);
          }}
          style={{ borderColor: cleared ? 'var(--status-valid)' : '', color: cleared ? 'var(--status-valid)' : '' }}
        >
          {cleared ? '✓ SERVER LOGS CLEARED' : 'PERMANENTLY CLEAR SERVER LOG DATA'}
        </button>
      </div>
    </div>
  );
}
