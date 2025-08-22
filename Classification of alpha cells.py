def calculate_auc_metrics_by_events(df, event_threshold=20, tolerance=0.15):
    ranges = [
        (0, 600, "leg_1"), (601, 1200, "leg_2"),
        (1201, 1800, "leg_3"), (1801, 2400, "leg_4"),
        (2401, 3000, "leg_5"), (3001, 3600, "leg_6")
    ]
    
    fractions = []
    
    for roi, group in df.groupby("roi"):
        leg_event_counts = {}
        total_events = 0
        inactive_condition = True

        for start, end, leg_name in ranges:
            leg_events = len(group[(group["peakpoint"] > start) & (group["peakpoint"] <= end)])
            leg_event_counts[leg_name] = leg_events
            total_events += leg_events
            if leg_events >= 5:
                inactive_condition = False

        max_events = max(leg_event_counts.values())
        min_events = min(leg_event_counts.values())
        relative_difference = abs(max_events - min_events) / total_events if total_events > 0 else 0

        if inactive_condition:
            activity_status = "inactive"
        elif relative_difference <= tolerance:
            activity_status = "constantly_active"
        else:
            largest_leg = max(leg_event_counts, key=leg_event_counts.get)
            activity_status = f"{largest_leg}_dominant"

        fractions.append({
            "roi": roi,
            "activity_status": activity_status,
            **leg_event_counts,
            "total_events": total_events,
        })

    return pd.DataFrame(fractions)


# Usage
fractions_df = calculate_auc_metrics_by_events(Events, event_threshold=300, tolerance=0.15)

# Streamlined summary table
activity_summary = fractions_df.groupby('activity_status').agg(
    ROIs=('roi', lambda x: ', '.join(map(str, x))),
    Total_ROIs=('roi', 'size')
).reset_index()

print(activity_summary)