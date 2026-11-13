# Operations and recovery

## Deployment

The retailer’s Kubernetes ingress terminates HTTPS and forwards traffic to the application’s ClusterIP Service. The Service routes to a ready API pod. Set the ingress read timeout above the 120-second report deadline. Each pod exposes internal metrics for scraping. Three CronJobs run the publication, preference and regression-sampling commands from the same image. [Infrastructure and ownership](HLD.md#assumptions)

Deploy the PII service behind its own internal ClusterIP. Mark each replica ready once its recognizers and the English spaCy model are loaded. API pods and workers that redact text call it within their operation deadline. Track its latency and timeouts when sizing replicas.

## Turn state and retries

The gateway records one turn in PostgreSQL under the employee, conversation and client request ID. The record holds the input digest and a stable `run_id`. If the client reconnects with the same key and input, return the existing run, preview or receipt after checking access. Reusing the key with different input is a conflict.

A replacement replica restores the LangGraph checkpoint, including the remaining budget and any pending decision. The SQL tool handler checks the original BigQuery job ID before submitting work again. The Reports Service checks the operation receipt before repeating a save or confirmed deletion. [Report transactions](reports.md#postgresql-implementation)

## Deadlines and failure handling

The gateway starts the clock at admission. For a chat message, the intent decision selects the task deadline and preserves time and cost already spent. Proposed defaults are 30 seconds for a question and 120 seconds for a requested report, with a configurable $1 budget per turn. Allow up to 12 model attempts and four SQL attempts in total, including at most two corrections for a failed SQL step and two transient model retries. Time spent waiting for deletion approval pauses the compute clock. The preview keeps its own expiry.

| Failure | Service response |
| --- | --- |
| Invalid SQL, unknown field or type error | Return a redacted diagnostic. A correction spends another attempt and goes through the same validation and access checks. |
| Empty result | Check the dates and filters. If they match the request, report no data. |
| Gemini rate limit or temporary outage | Retry with backoff and jitter within the remaining attempts, deadline and provider quota. |
| BigQuery denies access | Stop the query. |
| BigQuery response lost after submission | Look up the original job. Keep its budget reservation until billing is known. |
| Report commit response lost | Return the stored receipt for the operation. |
| Golden Retriever unavailable | Record the failed lookup. If the question or the employee’s preferences fully specify the calculation and the authorized schema supports it, proceed and state the definition. If the request depends on a particular Golden case, explain that its method could not be retrieved and ask for the missing details. |
| Identity, grants or privacy checks unavailable | Stop before sending protected data to the model or logs. |
| Only part of the analysis succeeds | Return the available query results and name the part that could not be completed. |

## Restore and release

The retailer sets retention and recovery targets for checkpoints, reports, evidence, Golden revisions and traces. In a restore drill, open a report with its original evidence, replay a committed deletion and verify an active Golden search entry against its bucket object. Recheck current grants before returning restored content.

After the evaluation gates pass, deploy the versioned image and configuration to staging. Exercise one turn against Gemini and BigQuery, then roll out to a limited share of traffic. Monitor task outcomes, latency and cost. The release owner rolls back a critical regression. [Evaluation and acceptance scenarios](evaluation-and-observability.md#acceptance-scenarios)
