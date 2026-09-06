from ai_agent.core.voice.classifier import TaskClassifier, TaskProfile, tokenize


def test_tokenize_drops_stopwords_and_short_tokens():
    tokens = tokenize("Это и то, а также бюджет на маркетинг в этом квартале")
    assert "бюджет" in tokens
    assert "маркетинг" in tokens
    assert "квартале" in tokens
    assert "и" not in tokens
    assert "а" not in tokens
    assert "то" not in tokens


def test_task_profile_confidence_starts_at_zero():
    profile = TaskProfile(task_id=1)
    assert profile.confidence == 0.0


def test_task_profile_confidence_reflects_confirm_reject_ratio():
    profile = TaskProfile(task_id=1)
    profile.apply_feedback(tokenize("бюджет квартал"), positive=True)
    profile.apply_feedback(tokenize("бюджет квартал"), positive=True)
    profile.apply_feedback(tokenize("бюджет квартал"), positive=False)
    assert abs(profile.confidence - 2 / 3) < 1e-9


def test_task_profile_round_trip_dict():
    profile = TaskProfile(task_id=7)
    profile.seed("бюджет согласование квартал")
    profile.apply_feedback(tokenize("бюджет утверждён"), positive=True)
    restored = TaskProfile.from_dict(profile.to_dict())
    assert restored.task_id == 7
    assert restored.positive_words == profile.positive_words
    assert restored.confirmations == 1


def test_classify_with_no_tasks_returns_unassigned():
    classifier = TaskClassifier()
    result = classifier.classify("Что-нибудь про бюджет")
    assert result.task_id is None
    assert result.auto is False
    assert result.needs_confirmation is False


def test_classify_empty_text_returns_unassigned():
    classifier = TaskClassifier()
    classifier.register_task(1, seed_text="бюджет квартал")
    result = classifier.classify("   ")
    assert result.task_id is None
    assert result.candidates == []


def test_classify_single_strongly_matching_task_is_automatic():
    classifier = TaskClassifier()
    classifier.register_task(1, seed_text="бюджет квартальный отчёт финансы")
    result = classifier.classify("Нужно согласовать квартальный бюджет и финансы")
    assert result.task_id == 1
    assert result.auto is True
    assert result.needs_confirmation is False


def test_classify_ambiguous_between_two_tasks_needs_confirmation():
    classifier = TaskClassifier()
    classifier.register_task(1, seed_text="бюджет финансы квартал")
    classifier.register_task(2, seed_text="маркетинг реклама квартал")
    # Реплика содержит равные по весу пересечения с обеими задачами.
    result = classifier.classify("квартал")
    assert result.needs_confirmation is True
    assert result.task_id is None
    assert {c[0] for c in result.candidates} == {1, 2}


def test_confirming_task_raises_confidence_and_reduces_future_questions():
    classifier = TaskClassifier()
    classifier.register_task(1, seed_text="проект альфа")
    classifier.register_task(2, seed_text="проект бета")

    ambiguous_text = "обсудили проект"
    first = classifier.classify(ambiguous_text)
    assert first.needs_confirmation is True

    # Пользователь несколько раз подтверждает, что подобные реплики — про задачу 1.
    for _ in range(6):
        classifier.confirm(1, "обсудили проект статус")

    assert classifier.profiles[1].confidence == 1.0

    second = classifier.classify(ambiguous_text)
    # После накопленных подтверждений классификатор увереннее в задаче 1 —
    # либо уже не переспрашивает, либо явно предпочитает её первым кандидатом.
    assert second.candidates[0][0] == 1
    if second.auto:
        assert second.task_id == 1


def test_rejecting_task_lowers_its_future_score():
    classifier = TaskClassifier()
    classifier.register_task(1, seed_text="бюджет отчёт")
    before = classifier.profiles[1].score(tokenize("бюджет отчёт квартальный"))

    classifier.reject(1, "бюджет отчёт квартальный")
    after = classifier.profiles[1].score(tokenize("бюджет отчёт квартальный"))

    assert after < before
