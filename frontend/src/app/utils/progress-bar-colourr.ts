export function getProgressBarColor(percent: number): string {
  if (percent >= 75) return 'success';
  if (percent >= 50) return 'warning';
  if (percent >= 25) return 'warning';
  return 'danger';
}
