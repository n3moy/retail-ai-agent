import asyncio
from collections.abc import Iterator
from contextlib import closing

from google.cloud import bigquery
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from retail_ai_agent.config import Config
from retail_ai_agent.infrastructure.local_state import LocalState
from retail_ai_agent.infrastructure.reports import FileReportStore
from retail_ai_agent.infrastructure.startup import startup_step


def report_store(state: LocalState) -> FileReportStore:
    with startup_step("Could not open report files. Check DATA_DIR/reports."):
        return FileReportStore(state.reports_path)


def bigquery_client(config: Config) -> Iterator[bigquery.Client]:
    with startup_step("Could not create the BigQuery client. Check GOOGLE_CLOUD_PROJECT and ADC."):
        client = bigquery.Client(project=config.google_cloud_project)
    with closing(client):
        yield client


def gemini_model(config: Config) -> Iterator[ChatGoogleGenerativeAI]:
    with startup_step("Could not create the Gemini client. Check the API key and GEMINI_MODEL."):
        assert config.google_ai_studio_api_key is not None
        model = ChatGoogleGenerativeAI(
            model=config.model_name,
            api_key=config.google_ai_studio_api_key.get_secret_value(),
            vertexai=False,
            max_retries=0,
            timeout=config.model_timeout_seconds,
            max_output_tokens=config.max_output_tokens,
        )
        if model.client is None:
            raise ValueError("Gemini client is not initialized")
    with closing(model.client):
        yield model


def openrouter_model(config: Config) -> Iterator[ChatOpenAI]:
    with startup_step(
        "Could not create the OpenRouter client. Check OPENROUTER_API_KEY, "
        "OPENROUTER_MODEL and OPENROUTER_BASE_URL."
    ):
        assert config.openrouter_api_key is not None
        model = ChatOpenAI(
            model=config.openrouter_model,
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            use_responses_api=False,
            max_retries=0,
            timeout=config.model_timeout_seconds,
            max_completion_tokens=config.max_output_tokens,
        )
    with closing(model.root_client):
        try:
            yield model
        finally:
            # The CLI is synchronous; ChatOpenAI also creates an async SDK client.
            asyncio.run(model.root_async_client.close())


def chat_model(config: Config) -> Iterator[BaseChatModel]:
    if config.model_provider == "openrouter":
        yield from openrouter_model(config)
    else:
        yield from gemini_model(config)
