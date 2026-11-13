from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.todo import PlanningState
from langgraph.runtime import Runtime

from retail_ai_agent.infrastructure.agent.context import RunContext


class SqlPlanningMiddleware(TodoListMiddleware):
    def __init__(self):
        super().__init__(
            system_prompt=(
                "Use write_todos for SQL analysis with dependent steps, such as comparing "
                "periods and then investigating the categories driving a change. Plan the "
                "metric, grain, joins, filters and result checks that matter to this question. "
                "Keep steps short. Revise the plan as schema or query results change it, and "
                "mark steps complete when supported by observations. Answer simple questions "
                "directly. If the user asks only for a plan, describe it in the final answer "
                "and leave data queries for later."
            ),
            tool_description=(
                "Replace the current SQL analysis plan with the full list of steps and their "
                "statuses. Use at most once per model response. Use sql_db_query for data."
            ),
        )

    def before_agent(self, state: PlanningState, runtime: Runtime[RunContext]) -> dict:
        # A new request starts here. Resuming an interrupted tool keeps the current plan.
        return {"todos": []}
