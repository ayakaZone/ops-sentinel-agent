"""全站每日调用次数计数器（内存实现，按天重置，用于软限流）"""

from datetime import date
from threading import Lock
from typing import Dict, Optional


class DailyUsageCounter:
    """按天、按分类统计调用次数的计数器"""

    def __init__(self):
        # key 格式："YYYY-MM-DD:分类名"，例如 "2026-07-21:chat"
        self._counts: Dict[str, int] = {}
        self._lock = Lock()

    def increment(self, category: str) -> int:
        """记一次调用，返回今天该分类累计的次数"""
        key = f"{date.today().isoformat()}:{category}"
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + 1
            return self._counts[key]

    def get_today_count(self, category: str) -> int:
        """只查询今天的次数，不增加计数"""
        key = f"{date.today().isoformat()}:{category}"
        with self._lock:
            return self._counts.get(key, 0)

    def increment_and_get_reminder(self, category: str, limit: int) -> Optional[str]:
        """记一次调用；若超过阈值，返回附加在回答末尾的软提醒文案，否则返回 None"""
        count = self.increment(category)
        if count > limit:
            return f"\n\n---\n💡 今日「{category}」调用量已达 {count} 次（阈值 {limit} 次），请注意合理使用。"
        return None


# 全局单例
daily_usage_counter = DailyUsageCounter()
