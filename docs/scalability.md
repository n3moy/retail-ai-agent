# Extending and scaling the assistant

## Adding an analytical source

The agent keeps calling `execute_analysis_sql(sql)`. The SQL tool handler selects the registered `AnalyticsSource` adapter that serves the authorized relations in the query. Adding a warehouse such as PostgreSQL or Snowflake involves three steps.

1. Register its read-only connection and expose its permitted relations, columns and joins. Enforce the employee’s brand and column permissions in the database or its approved views.
2. Configure SQL validation for that dialect. Implement submission, cancellation, timeout, cost and result limits using the source’s own controls. BigQuery’s `maximum_bytes_billed` applies to BigQuery jobs.
3. Return the same fields to the agent: `status`, `rows`, `columns`, `query_id`, `source`, `units`, `period` and `warnings`. Test expected figures, unauthorized access, query failures and disclosure on a source-specific fixture.

For a cross-source analysis, query each warehouse under the employee’s grants, then join the authorized aggregates on agreed keys. A recurring cross-source workload may justify a dedicated data mart.

## Adding a tool

Register a typed tool in LangChain, implement its application handler and add evaluation cases for permissions, failures and retries. The gateway supplies employee identity and brand grants. Each handler checks access when called.

| Tool | Data it uses and checks it makes |
| --- | --- |
| `create_chart(evidence_id, chart_type, dimensions)` | Reads a query result released for the employee’s current brand grants. Restricts chart types and row counts before rendering. |
| `send_report(report_id, recipient)` | The Reports Service checks owner and recipient access. Require explicit approval before sending through the retailer’s mail provider and deduplicate retries. |
| `search_public_web(query)` | Sends only public search terms. Treat retrieved pages as untrusted references and cite external claims. |

With a small catalog, the ReAct loop can use all tools allowed for the turn. If tool descriptions start consuming too much context or evaluations show the agent choosing the wrong tools, add a selection step before the first ReAct model call. It matches the request and conversation context against the permitted tool descriptions and binds the selected schemas for that turn. For “chart monthly sales,” it would select both `execute_analysis_sql` and `create_chart` so the agent can query the figures and pass the resulting evidence ID to the chart tool.

Review tools that send data outside the retailer before enabling them. Roll out a new integration to a limited group and compare figures, errors, latency and cost. Existing reports retain their IDs, conversation references and original calculations across deployments.

## Scaling the existing services

The initial load test assumes one incoming request per second, including concurrent reports. Measure time and saturation at each component before changing capacity.

| If the bottleneck is… | First change |
| --- | --- |
| Application replicas | Add replicas that share the LangGraph PostgreSQL checkpointer, Store and Reports Service records. Any ready replica can resume a turn. |
| Gemini quota | Limit concurrent calls and retries across replicas. A process-local semaphore protects only that process. Delay or reject excess work before requesting more quota. |
| BigQuery jobs | Limit concurrent jobs, scanned bytes and elapsed time. Optimize repeated expensive queries or use approved pre-aggregations. |
| Long reports | If generation regularly exceeds the 120-second budget, move it to a resumable job and return its ID. |
| Saved report search | Start with PostgreSQL full-text search and a GIN index behind `search_reports(query)`. If searches are too slow or relevant reports rarely rank near the top on representative queries, evaluate Elasticsearch for ranking candidate IDs. Before returning even a title, the Reports Service verifies in PostgreSQL that the report is active, belongs to the employee and falls within their current brand grants. |
| Slow deletion previews | Narrow candidate rows by owner, current grants and conversation before matching the phrase. If scans still miss the latency target, index candidate selection and verify every match before preview. |
