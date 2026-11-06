"""CLI tool for running offline or scheduled model monitoring cycles (AC7, AC8).

Executes:
1. Production prediction-to-outcome correlation
2. Data quality filtering and exclusion
3. Feature drift detection (PSI) across the 6 canonical features
4. Outcome-based performance evaluation (MAE, RMSE)
5. Retraining trigger evaluation and candidate generation
"""

import argparse
import asyncio
from datetime import datetime, timezone
import sys

from app.application.monitoring.service import run_monitoring_cycle
from app.infrastructure.config.settings import settings
from app.infrastructure.database.mongo_client import mongo_manager
from app.infrastructure.ml_models.loader import model_loader


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="SpotQ ETA Production Model Monitoring & Drift Detection Cycle"
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Hours of historical data to evaluate (default: 24)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help=f"Minimum required valid samples for evaluation (default: {settings.MONITORING_MIN_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--restaurant-id",
        type=str,
        default=None,
        help="Optional restaurant identifier filter",
    )
    parser.add_argument(
        "--auto-trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to trigger candidate retraining if degradation or drift is detected (default: True)",
    )
    parser.add_argument(
        "--dataset-mode",
        choices=["synthetic", "production"],
        default="synthetic",
        help="Dataset mode to use if candidate retraining is triggered (default: synthetic)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SPOTQ ETA PRODUCTION MODEL MONITORING & RETRAINING EVALUATION")
    print("=" * 70)
    print(f"Timestamp        : {datetime.now(timezone.utc).isoformat()}")
    print(f"Lookback Window  : {args.lookback_hours} hours")
    print(f"Auto-Trigger     : {args.auto_trigger}")
    print(f"Dataset Mode     : {args.dataset_mode}")
    print("-" * 70)

    # 1. Connect MongoDB
    print("\n1. Connecting to MongoDB...")
    await mongo_manager.connect()
    print(f"   Connected to database: {settings.MONGO_DB_NAME}")

    # 2. Ensure model loader snapshot is available
    if not model_loader.is_healthy():
        print("   Loading active ETA model snapshot...")
        model_loader.load_model()

    active_snap = model_loader.get_snapshot()
    active_version = getattr(active_snap, "model_version", "unknown")
    print(f"   Active model version: {active_version}")

    # 3. Run monitoring cycle
    print("\n2. Executing monitoring cycle...")
    try:
        report = await run_monitoring_cycle(
            lookback_hours=args.lookback_hours,
            min_sample_size=args.min_samples,
            restaurant_id=args.restaurant_id,
            auto_trigger=args.auto_trigger,
            dataset_mode=args.dataset_mode,
        )
    except Exception as exc:
        print(f"\n❌ Error executing monitoring cycle: {exc}", file=sys.stderr)
        await mongo_manager.disconnect()
        return 1

    # 4. Display Results
    print("\n3. Prediction-to-Outcome Correlation:")
    corr = report.correlation_summary
    print(f"   - Total predictions in window : {corr.total_predictions}")
    print(f"   - Completed outcomes in window: {corr.total_completed_outcomes}")
    print(f"   - Correlated records          : {corr.correlated_count}")
    print(f"   - Uncorrelated predictions    : {corr.uncorrelated_predictions_count}")
    print(f"   - Uncorrelated outcomes       : {corr.uncorrelated_outcomes_count}")

    print("\n4. Data Quality & Exclusion Guard:")
    dq = report.data_quality_report
    print(f"   - Total evaluated records     : {dq.total_evaluated}")
    print(f"   - Valid records               : {dq.valid_count}")
    print(f"   - Excluded records            : {dq.excluded_count}")
    print(f"   - Sufficiency for evaluation  : {'YES' if dq.is_sufficient_for_evaluation else 'NO'}")
    if dq.excluded_count > 0:
        rej = dq.rejections
        print("   - Exclusions breakdown:")
        print(f"       * Missing features        : {rej.missing_features}")
        print(f"       * Non-finite features     : {rej.non_finite_features}")
        print(f"       * Invalid target wait     : {rej.invalid_target_wait}")
        print(f"       * Duplicate queue entries : {rej.duplicate_queue_entry}")
        print(f"       * Invalid timestamps      : {rej.invalid_timestamps}")

    print("\n5. Feature Drift Detection (PSI):")
    drift = report.drift_report
    print(f"   - Baseline reference version  : {drift.baseline_version}")
    print(f"   - Significant drift detected  : {'YES' if drift.has_significant_drift else 'NO'}")
    print(f"   - Drifted features count      : {drift.drifted_features_count}/6")
    for feat_name, f_res in drift.feature_results.items():
        drift_flag = "⚠️ DRIFT" if f_res.is_drifted else "✅ OK"
        print(f"       * {feat_name:24}: PSI = {f_res.psi_score:6.3f} [{f_res.drift_level:18}] {drift_flag}")

    print("\n6. Model Accuracy & Error Performance:")
    perf = report.performance_report
    print(f"   - Overall MAE                 : {perf.overall_mae_minutes:.2f} min" if perf.overall_mae_minutes else "   - Overall MAE                 : N/A")
    print(f"   - Overall RMSE                : {perf.overall_rmse_minutes:.2f} min" if perf.overall_rmse_minutes else "   - Overall RMSE                : N/A")
    print(f"   - Degradation detected        : {'YES' if perf.has_degradation else 'NO'}")
    for ver, vperf in perf.version_metrics.items():
        status_flag = "❌ DEGRADED" if vperf.is_degraded else "✅ HEALTHY"
        mae_str = f"{vperf.mae_minutes:.2f} min" if vperf.mae_minutes is not None else "N/A"
        rmse_str = f"{vperf.rmse_minutes:.2f} min" if vperf.rmse_minutes is not None else "N/A"
        print(f"       * Version [{ver}]: N={vperf.sample_count}, MAE={mae_str}, RMSE={rmse_str} -> {status_flag}")
        if vperf.degradation_reasons:
            for reason in vperf.degradation_reasons:
                print(f"           - {reason}")

    print("\n7. Retraining Trigger Decision (AC5, AC6):")
    dec = report.trigger_decision
    if dec is not None:
        print(f"   - Retrain Requested           : {'YES' if dec.should_retrain else 'NO'}")
        print(f"   - Trigger Type                : {dec.trigger_type}")
        print(f"   - Decision Status             : {dec.status}")
        print(f"   - Trigger Reason              : {dec.reason}")
    else:
        print("   - Auto-triggers disabled for this cycle.")

    if report.retraining_execution is not None:
        retrain = report.retraining_execution
        print("\n8. Retraining Execution Outcome (AC6 Safety Boundary):")
        print(f"   - Status                      : {retrain.status}")
        print(f"   - Candidate Artifact ID       : {retrain.candidate_artifact_id or 'None'}")
        print(f"   - Eligible for Promotion      : {retrain.is_eligible_for_promotion}")
        print("   - Note: Model was generated as candidate only; NOT auto-promoted.")

    print("\n" + "=" * 70)
    print("CYCLE COMPLETE")
    print("=" * 70)

    # Disconnect MongoDB
    await mongo_manager.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
