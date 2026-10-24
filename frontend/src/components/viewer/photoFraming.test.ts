import { expect, test } from 'vitest';
import { photoVerticalFov } from './photoFraming';
test('contains the entire photo without narrowing the source vertical field', () => {
  for (const aspect of [0.6, 1.5, 2.1]) {
    const vertical = photoVerticalFov(1.15, 1.5, aspect) * Math.PI / 180;
    expect(Math.tan(vertical / 2)).toBeGreaterThanOrEqual(0.5 / 1.15 - 1e-12);
    expect(Math.tan(vertical / 2) * aspect).toBeGreaterThanOrEqual(0.5 / 1.15 * 1.5 - 1e-12);
  }
});
test('restores the original vertical lens on widescreen', () => {
  expect(photoVerticalFov(1.15, 1.5, 2.1)).toBeCloseTo(2 * Math.atan(0.5 / 1.15) * 180 / Math.PI, 10);
});
