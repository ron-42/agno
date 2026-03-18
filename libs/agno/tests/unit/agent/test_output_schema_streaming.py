from typing import Any, AsyncIterator, Iterator

import pytest
from pydantic import BaseModel

from agno.agent import Agent
from agno.models.base import Model
from agno.models.response import ModelResponse


class StructuredResponse(BaseModel):
    answer: str


class SpyStreamingModel(Model):
    def __init__(self):
        super().__init__(id="spy-stream-model", name="SpyStreamModel", provider="test")
        self.invoke_calls = 0
        self.invoke_stream_calls = 0
        self.ainvoke_calls = 0
        self.ainvoke_stream_calls = 0

    def invoke(self, *args, **kwargs) -> ModelResponse:
        self.invoke_calls += 1
        return ModelResponse(role="assistant", content='{"answer":"non-stream"}')

    async def ainvoke(self, *args, **kwargs) -> ModelResponse:
        self.ainvoke_calls += 1
        return ModelResponse(role="assistant", content='{"answer":"non-stream"}')

    def invoke_stream(self, *args, **kwargs) -> Iterator[ModelResponse]:
        self.invoke_stream_calls += 1
        yield ModelResponse(role="assistant", content='{"answer":"st')
        yield ModelResponse(role="assistant", content='ream"}')

    async def ainvoke_stream(self, *args, **kwargs) -> AsyncIterator[ModelResponse]:
        self.ainvoke_stream_calls += 1
        yield ModelResponse(role="assistant", content='{"answer":"st')
        yield ModelResponse(role="assistant", content='ream"}')

    def _parse_provider_response(self, response: Any, **kwargs) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


def test_output_schema_streaming_uses_provider_streaming_sync(caplog):
    model = SpyStreamingModel()
    agent = Agent(model=model, output_schema=StructuredResponse, telemetry=False)

    final_run_output = None
    for item in agent.run("Return JSON", stream=True, yield_run_output=True):
        if hasattr(item, "content_type"):
            final_run_output = item

    assert final_run_output is not None
    assert isinstance(final_run_output.content, StructuredResponse)
    assert final_run_output.content.answer == "stream"
    assert model.invoke_stream_calls == 1
    assert model.invoke_calls == 0
    assert "Failed to convert response to output_schema" not in caplog.text
    assert "All parsing attempts failed." not in caplog.text


@pytest.mark.asyncio
async def test_output_schema_streaming_uses_provider_streaming_async(caplog):
    model = SpyStreamingModel()
    agent = Agent(model=model, output_schema=StructuredResponse, telemetry=False)

    final_run_output = None
    async for item in agent.arun("Return JSON", stream=True, stream_events=True, yield_run_output=True):
        if hasattr(item, "content_type"):
            final_run_output = item

    assert final_run_output is not None
    assert isinstance(final_run_output.content, StructuredResponse)
    assert final_run_output.content.answer == "stream"
    assert model.ainvoke_stream_calls == 1
    assert model.ainvoke_calls == 0
    assert "Failed to convert response to output_schema" not in caplog.text
    assert "All parsing attempts failed." not in caplog.text
