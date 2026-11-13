from dataclasses import dataclass

from dishka import Container, Provider, Scope, alias, from_context, make_container, provide
from google.cloud import bigquery
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from retail_ai_agent.application.access import allowed_brands
from retail_ai_agent.application.chat import ChatService
from retail_ai_agent.application.ports import AgentRunner, Analytics, ReportStore
from retail_ai_agent.application.reports import ReportService
from retail_ai_agent.config import Config
from retail_ai_agent.domain.models import Actor
from retail_ai_agent.infrastructure import resources
from retail_ai_agent.infrastructure.agent.context import RunContextFactory
from retail_ai_agent.infrastructure.agent.graph import create_chat_graph
from retail_ai_agent.infrastructure.agent.runner import LangGraphRunner
from retail_ai_agent.infrastructure.bigquery import BigQueryAnalytics
from retail_ai_agent.infrastructure.local_state import LocalState
from retail_ai_agent.infrastructure.privacy import LocalSanitizer
from retail_ai_agent.infrastructure.reports import FileReportStore
from retail_ai_agent.infrastructure.schema import BigQuerySchemaLoader, SchemaCatalog
from retail_ai_agent.infrastructure.sql import SafeSql
from retail_ai_agent.infrastructure.startup import StartupChecks
from retail_ai_agent.infrastructure.tracing import JsonlTrace
from retail_ai_agent.infrastructure.users import UserDirectory


@dataclass(frozen=True)
class UserSelection:
    user_id: str


class AppProvider(Provider):
    # One CLI process serves one selected actor, so all dependencies live for the app.
    scope = Scope.APP
    config = from_context(Config)
    user_selection = from_context(UserSelection)

    @provide
    def users(self, config: Config) -> UserDirectory:
        return UserDirectory(config.users_file)

    @provide
    def actor(self, users: UserDirectory, selection: UserSelection) -> Actor:
        return users.actor(selection.user_id)

    @provide
    def state(self, config: Config) -> LocalState:
        return LocalState(config.data_dir)

    catalog = provide(SchemaCatalog)
    schema_loader = provide(BigQuerySchemaLoader)

    @provide
    def sql(self, config: Config, actor: Actor, catalog: SchemaCatalog) -> SafeSql:
        return SafeSql(config.bigquery_dataset, allowed_brands(actor), catalog)

    @provide
    def sanitizer(self, state: LocalState, users: UserDirectory) -> LocalSanitizer:
        return LocalSanitizer(pii_key=state.pseudonym_key(), brands=users.catalog_brands())

    @provide
    def trace(self, state: LocalState) -> JsonlTrace:
        return JsonlTrace(state.trace_path)

    @provide
    def checkpointer(self) -> BaseCheckpointSaver:
        return InMemorySaver()

    store = provide(staticmethod(resources.report_store))
    client = provide(staticmethod(resources.bigquery_client))
    model = provide(staticmethod(resources.chat_model))

    analytics = provide(BigQueryAnalytics)
    reports = provide(ReportService)
    contexts = provide(RunContextFactory)
    graph = provide(staticmethod(create_chat_graph))
    runner = provide(LangGraphRunner)
    chat = provide(ChatService)

    report_store_port = alias(FileReportStore, provides=ReportStore)
    analytics_port = alias(BigQueryAnalytics, provides=Analytics)
    runner_port = alias(LangGraphRunner, provides=AgentRunner)

    @provide
    def startup_checks(
        self,
        config: Config,
        sql: SafeSql,
        store: FileReportStore,
        trace: JsonlTrace,
        client: bigquery.Client,
        schema_loader: BigQuerySchemaLoader,
    ) -> StartupChecks:
        return StartupChecks(
            client=client,
            sql=sql,
            store=store,
            trace=trace,
            timeout=config.startup_timeout_seconds,
            schema_loader=schema_loader,
        )


def make_app_container(config: Config, user_id: str, *overrides: Provider) -> Container:
    """Validate the graph without opening resources; apply test providers before building."""
    return make_container(
        AppProvider(),
        *overrides,
        context={Config: config, UserSelection: UserSelection(user_id)},
    )
