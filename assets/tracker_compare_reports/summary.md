# ByteTrack vs BoT-SORT Comparison Summary

## Result

**Conclusion:** ByteTrack is faster, while BoT-SORT does not reduce raw ID count in this run.

| Metric | ByteTrack | BoT-SORT | Meaning |
|---|---:|---:|---|
| Total raw IDs | 13 | 13 | Lower is better |
| Short tracks | 2 | 2 | Lower is better |
| Average FPS | 16.7 | 10.9 | Higher is better |
| Average latency | 59.8 ms | 92.1 ms | Lower is better |

## Files

- Chart: `/mnt/nvme/opt/Tracking_research/assets/tracker_compare_reports/comparison_chart.png`
- Output video: `/mnt/nvme/opt/Tracking_research/assets/video_results/difference_bytetrack_vs_botsort_19.mp4`
- Detailed CSV: `/mnt/nvme/opt/Tracking_research/assets/tracker_compare_reports/metrics.csv`
- Summary JSON: `/mnt/nvme/opt/Tracking_research/assets/tracker_compare_reports/summary.json`

Note: true tracking accuracy requires manually labeled ground truth. Without ground truth, this report uses simple indicators: raw ID count, short tracks, and FPS.
