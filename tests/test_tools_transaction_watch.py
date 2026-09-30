from ffapp.tools.transaction_watch import completed_new_transactions, format_transaction


def test_new_transactions_are_completed_unseen_and_chronological() -> None:
    rows = [
        {"transaction_id": "newer", "status": "complete", "created": 20},
        {"transaction_id": "pending", "status": "pending", "created": 5},
        {"transaction_id": "seen", "status": "complete", "created": 1},
        {"transaction_id": "older", "status": "complete", "created": 10},
    ]
    result = completed_new_transactions(rows, {"seen"})
    assert [row["transaction_id"] for row in result] == ["older", "newer"]


def test_transaction_message_marks_user_and_explains_dropped_player() -> None:
    message = format_transaction(
        {
            "type": "free_agent",
            "adds": {"1": 2},
            "drops": {"2": 2},
            "roster_ids": [2],
        },
        {"1": {"full_name": "Added Player"}, "2": {"full_name": "Dropped Player"}},
        {2: "My Team"},
        2,
    )
    assert "My Team" in message
    assert "(you)" in message
    assert "Added Player" in message
    assert "Dropped Player" in message
    assert "Recheck the lineup" in message
