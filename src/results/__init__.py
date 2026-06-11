"""结果管理包导出列表和删除能力。 / Result-management package exports listing and deletion capabilities."""

from results.management import (
    CompactManagedLogsSummary,
    CompactSystemLogsSummary,
    DeleteManagedResultsSummary,
    LatestPromotionSummary,
    ManagedResult,
    compact_managed_result_logs,
    compact_system_logs,
    delete_managed_results,
    list_managed_results,
    promote_training_result_to_latest,
)

__all__ = [
    "CompactManagedLogsSummary",
    "CompactSystemLogsSummary",
    "DeleteManagedResultsSummary",
    "LatestPromotionSummary",
    "ManagedResult",
    "compact_managed_result_logs",
    "compact_system_logs",
    "delete_managed_results",
    "list_managed_results",
    "promote_training_result_to_latest",
]
