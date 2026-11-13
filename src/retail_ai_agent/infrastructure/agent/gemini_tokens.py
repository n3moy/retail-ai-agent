"""Count the complete Gemini request using the provider's tokenizer."""

import httpx
from langchain.agents.middleware import ModelRequest
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from retail_ai_agent.application.errors import ServiceUnavailable
from retail_ai_agent.infrastructure.agent.context import RunContext


def count_gemini_input(request: ModelRequest[RunContext], key: SecretStr, timeout: float) -> int:
    model = request.model
    if not isinstance(model, ChatGoogleGenerativeAI):
        raise ServiceUnavailable("The Gemini token counter requires a Gemini model.")
    messages = [request.system_message] if request.system_message else []
    # Reuse the adapter's generation serializer, including tool schemas and thought signatures.
    bound = model.bind_tools(request.tools).kwargs if request.tools else {}
    prepared = model._prepare_request([*messages, *request.messages], **bound)
    name = "models/" + model.model.removeprefix("models/")
    body = prepared["config"].model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        include={"system_instruction", "tools", "tool_config", "cached_content"},
    )
    body["model"] = name
    body["contents"] = [
        c.model_dump(mode="json", by_alias=True, exclude_none=True, warnings=False)
        for c in prepared["contents"]
    ]
    # The Developer API accepts tools via the REST generateContentRequest form.
    response = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/{name}:countTokens",
        headers={"x-goog-api-key": key.get_secret_value()},
        json={"generateContentRequest": body},
        timeout=timeout,
    )
    response.raise_for_status()
    total = response.json().get("totalTokens")
    if type(total) is not int or total < 0:
        raise ServiceUnavailable("Google did not return an input token count. Try again later.")
    return total
