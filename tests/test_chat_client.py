from unittest.mock import MagicMock, patch

import pytest
import requests

from ui.chat_client import send_chat_message


@patch("ui.chat_client.requests.post")
def test_send_chat_message_returns_answer_text(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"answer": "The average is 4.2 days."}
    mock_post.return_value = mock_response

    result = send_chat_message("what's the average time in hospital?", history=[])
    assert result == "The average is 4.2 days."


@patch("ui.chat_client.requests.post")
def test_send_chat_message_sends_message_and_history_as_json(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"answer": "ok"}
    mock_post.return_value = mock_response

    history = [{"role": "user", "content": "earlier question"}]
    send_chat_message("follow up question", history=history)

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["message"] == "follow up question"
    assert kwargs["json"]["history"] == history


@patch("ui.chat_client.requests.post")
def test_send_chat_message_raises_on_http_error(mock_post):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
    mock_post.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        send_chat_message("trigger a backend error", history=[])
