import React from 'react';

/** Keep a failed GLB/renderer from taking down the upload and Recent pages. */
export class ViewerErrorBoundary extends React.Component<React.PropsWithChildren<{ onRetry: () => void }>, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    if (this.state.failed) return (
      <div role="alert" className="absolute inset-0 z-50 flex flex-col items-center justify-center gap-4 bg-[#07070a] text-white">
        <p>The saved 3D model could not be loaded. Your reconstruction is still saved.</p>
        <button className="rounded bg-cyan-800 px-4 py-2" onClick={this.props.onRetry}>Retry loading model</button>
      </div>
    );
    return this.props.children;
  }
}
