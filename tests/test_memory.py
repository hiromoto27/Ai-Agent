from pathlib import Path

from ai_agent.core.memory import MemoryStore


def test_add_and_recent(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_episode(task="Создать отчёт в Word", outcome="Готово", success=True)
    store.add_episode(task="Посчитать бюджет в Excel", outcome="Ошибка формулы", success=False)

    recent = store.recent(limit=10)
    assert len(recent) == 2
    assert recent[0].task == "Посчитать бюджет в Excel"  # последний первый


def test_retrieve_similar_ranks_relevant_episode_first(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_episode(
        task="Создать презентацию про продажи за квартал",
        outcome="Готово, 8 слайдов",
        success=True,
        reflection="Использовать шаблон с графиками для отчётов о продажах",
    )
    store.add_episode(task="Полить цветы по расписанию", outcome="Готово", success=True)
    store.add_episode(task="Скачать курс валют с сайта ЦБ", outcome="Готово", success=True)

    results = store.retrieve_similar("Сделать презентацию по продажам за месяц", k=2)
    assert len(results) >= 1
    assert "продаж" in results[0].task.lower()
    assert "графиками" in results[0].reflection


def test_retrieve_similar_no_matches_returns_empty(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_episode(task="абвгд ежзик", outcome="x")
    results = store.retrieve_similar("совершенно другой запрос без общих слов", k=3)
    assert results == []


def test_stats_success_rate(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.add_episode(task="a", success=True)
    store.add_episode(task="b", success=True)
    store.add_episode(task="c", success=False)

    stats = store.stats()
    assert stats["total_episodes"] == 3
    assert stats["successes"] == 2
    assert stats["failures"] == 1
    assert abs(stats["success_rate"] - 2 / 3) < 1e-9


def test_stats_empty_store(tmp_path: Path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    stats = store.stats()
    assert stats["total_episodes"] == 0
    assert stats["success_rate"] is None


def test_persistence_across_reopen(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite3"
    store1 = MemoryStore(db_path)
    store1.add_episode(task="персистентная задача", success=True)
    store1.close()

    store2 = MemoryStore(db_path)
    recent = store2.recent(limit=5)
    assert len(recent) == 1
    assert recent[0].task == "персистентная задача"
