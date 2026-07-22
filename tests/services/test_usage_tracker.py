"""usage_tracker.DailyUsageCounter 的单元测试

涉及"按日期计数"，用 mocker 控制 date.today() 的返回值来测试跨天场景。
"""

import threading

from app.services.usage_tracker import DailyUsageCounter


def test_increment_returns_running_count():
    """每次 increment 返回的是累计到目前为止的次数"""
    counter = DailyUsageCounter()
    assert counter.increment("chat") == 1
    assert counter.increment("chat") == 2
    assert counter.increment("chat") == 3


def test_get_today_count_does_not_increment():
    """get_today_count 只读不写，不应该影响计数"""
    counter = DailyUsageCounter()
    counter.increment("chat")
    counter.increment("chat")
    assert counter.get_today_count("chat") == 2
    assert counter.get_today_count("chat") == 2  # 再查一次，数字不变


def test_categories_are_counted_independently():
    """chat 和 aiops 两个分类的计数互不影响"""
    counter = DailyUsageCounter()
    counter.increment("chat")
    counter.increment("chat")
    counter.increment("aiops")
    assert counter.get_today_count("chat") == 2
    assert counter.get_today_count("aiops") == 1


def test_count_resets_on_a_new_day(mocker):
    """
    跨天后计数器应该重新从 0 开始——key 里带着日期字符串，
    换了一天等于换了一把新 key，天然不用专门写"清零"逻辑。
    """
    counter = DailyUsageCounter()

    # 把 usage_tracker 模块里引用的 date 整体替换成假对象，控制 today() 的返回值
    mock_date = mocker.patch("app.services.usage_tracker.date")
    mock_date.today.return_value.isoformat.return_value = "2026-07-01"

    counter.increment("chat")
    counter.increment("chat")
    assert counter.get_today_count("chat") == 2

    # 模拟时间来到第二天
    mock_date.today.return_value.isoformat.return_value = "2026-07-02"
    assert counter.get_today_count("chat") == 0  # 新的一天，计数从 0 开始

    counter.increment("chat")
    assert counter.get_today_count("chat") == 1


def test_increment_and_get_reminder_below_limit_returns_none():
    """未超过阈值时不应该附加提示文案"""
    counter = DailyUsageCounter()
    for _ in range(5):
        reminder = counter.increment_and_get_reminder("chat", limit=10)
    assert reminder is None


def test_increment_and_get_reminder_above_limit_returns_text():
    """超过阈值后应该返回附加提示文案，且文案里包含分类名和当前次数"""
    counter = DailyUsageCounter()
    reminder = None
    for _ in range(11):  # limit=10，第 11 次调用超过阈值
        reminder = counter.increment_and_get_reminder("chat", limit=10)
    assert reminder is not None
    assert "chat" in reminder
    assert "11" in reminder


def test_increment_is_thread_safe():
    """
    并发调用 increment 时不应该发生计数丢失（验证 Lock 真的生效）。
    起 50 个线程各调用 20 次 increment，最终总数必须精确等于 1000，
    如果 Lock 失效，多线程同时读写共享字典会导致部分自增被覆盖丢失。
    """
    counter = DailyUsageCounter()
    thread_count, increments_per_thread = 50, 20

    def worker():
        for _ in range(increments_per_thread):
            counter.increment("chat")

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.get_today_count("chat") == thread_count * increments_per_thread
