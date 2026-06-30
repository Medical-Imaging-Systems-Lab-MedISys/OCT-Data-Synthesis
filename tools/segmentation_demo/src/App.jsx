import React from 'react'
import './App.css'

function App() {
  return (
    <div style={{ padding: '20px', fontFamily: 'sans-serif', maxWidth: '800px', margin: '0 auto', background: '#f9f9f9', borderRadius: '10px' }}>
      <h1 style={{ color: '#333' }}>OCT Segmentation Model: Inputs & Masks</h1>
      
      <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)', marginBottom: '20px' }}>
        <h2 style={{ color: '#2c3e50' }}>1. The Input (OCT Image)</h2>
        <p><strong>Format:</strong> Typically a 2D grayscale B-scan image (e.g., 256x256 or 512x512 pixels).</p>
        <p><strong>Purpose:</strong> This is the raw optical coherence tomography scan showing the cross-sectional view of the retina. The model takes this as its primary input to analyze structural features and abnormalities.</p>
        <p><strong>Preprocessing:</strong> Images are usually normalized (pixel values mapped to [0, 1] or [-1, 1]) and sometimes enhanced to remove speckle noise before being fed into a convolutional neural network (like a U-Net).</p>
      </div>

      <div style={{ background: '#fff', padding: '20px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
        <h2 style={{ color: '#2c3e50' }}>2. The Target Mask (Segmentation Map)</h2>
        <p><strong>Format:</strong> A 2D single-channel image with the exact same dimensions as the input. Instead of intensity values, each pixel contains a discrete <strong>integer class label</strong>.</p>
        <p><strong>Purpose:</strong> This acts as the "ground truth" during training. The model learns to predict this exact map. For example, in the OCT5k dataset, the classes represent:</p>
        <ul>
          <li><strong>Class 0:</strong> Background</li>
          <li><strong>Class 1:</strong> ILM (Inner Limiting Membrane)</li>
          <li><strong>Class 2:</strong> OPL-Henles</li>
          <li><strong>Class 3:</strong> IS/OS Junction</li>
          <li><strong>Class 4:</strong> IBRPE (Inner Boundary of Retinal Pigment Epithelium)</li>
          <li><strong>Class 5:</strong> OBRPE (Outer Boundary of Retinal Pigment Epithelium)</li>
        </ul>
        <p><strong>Visual Appearance:</strong> Because the values are 0, 1, 2, 3, 4, 5, the image appears pitch black to the human eye when opened in a standard viewer. In training code, these masks are often converted into one-hot tensors or passed directly into categorical loss functions like CrossEntropyLoss.</p>
      </div>
    </div>
  )
}

export default App
