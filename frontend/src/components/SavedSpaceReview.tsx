import React from 'react';
import type { SpaceJob } from '../types';

export const SavedSpaceReview: React.FC<{ space: SpaceJob }> = ({ space }) => {
  const result = space.gaussian;
  if (space.status === 'review_required' || result?.requires_review || result?.quality_accepted === false) {
    return <small data-quality="review-required" className="text-amber-200">Review required · quality not accepted</small>;
  }
  if (result?.experimental) return <small data-quality="experimental">Experimental reconstruction</small>;
  // A legacy "completed" state means processing ended, not that quality passed.
  return null;
};
