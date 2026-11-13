from retail_ai_agent.application.errors import InvalidRequest
from retail_ai_agent.application.models import ChatRequest, ChatResponse
from retail_ai_agent.application.ports import AgentRunner


class ChatService:
    def __init__(self, runner: AgentRunner):
        self.runner = runner

    def ask(self, request: ChatRequest) -> ChatResponse:
        if not request.conversation_id.strip():
            # TODO: When could the conversation ID be missing?
            # Could this check move to model validation?
            # The CLI should always have a conversation ID. It changes when a new CLI
            # process starts or the /new command is entered.
            raise InvalidRequest("Conversation ID is expected.")
        if not request.question.strip():
            raise InvalidRequest("Please provide your question.")
        if len(request.question) > 8000:
            raise InvalidRequest("Shorten your question to at most 8000 characters.")
        return self.runner.run(request)

    def confirm(self, conversation_id: str, action_id: str, approved: bool) -> ChatResponse:
        if type(approved) is not bool or not action_id or not conversation_id:
            raise InvalidRequest(
                "An action ID and explicit confirmation or cancellation are required."
            )
        return self.runner.resume(conversation_id, action_id, approved)

    def reselect(self, conversation_id: str, action_id: str, ids: list[str]) -> ChatResponse:
        return self.runner.reselect(conversation_id, action_id, ids)
