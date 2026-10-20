# MERN–Python ETA Queue Integration Contract

## Queue configuration

- Queue name: `ml-eta-queue`
- BullMQ prefix: `bull`
- Redis server: Redis 7.x
- Python BullMQ dependency: `bullmq==3.3.4`
- Python worker concurrency: 5 by default; configurable from 1 to 50.
- MongoDB database: configured by `MONGO_DB_NAME`
- MongoDB collection: `queue_events`

The Python worker and MERN producer must use the same Redis deployment,
queue name, and prefix. Redis credentials must be configured through
environment variables/secrets rather than committed to source control.

## Producer job options

Apply these options when adding jobs from MERN:

- attempts: 5 total attempts, including the initial attempt
- backoff: exponential, starting at 1,000 ms
- completed-job retention: up to 1,000 jobs and 24 hours
- failed-job retention: up to 1,000 jobs and 7 days

These are the agreed initial defaults for implementation. Confirm
operational retention requirements before production rollout.

## Event payload

Each job's data must be a JSON object containing:

- `queue_entry_id`: non-empty string, maximum 128 characters
- `restaurant_id`: non-empty string, maximum 128 characters
- `party_size`: integer from 1 to 100
- `joined_at`: timezone-aware ISO 8601 timestamp
- `completed_at`: timezone-aware ISO 8601 timestamp, later than joined_at
- `actual_wait_time_minutes`: finite number greater than 0 and at most 120
- `total_people_ahead_at_join`: integer from 0 to 1,000

Example:

{
  "queue_entry_id": "queue-entry-123",
  "restaurant_id": "restaurant-456",
  "party_size": 2,
  "joined_at": "2026-10-19T10:00:00Z",
  "completed_at": "2026-10-19T10:12:00Z",
  "actual_wait_time_minutes": 12,
  "total_people_ahead_at_join": 5
}

The reported wait duration must match the timestamp difference within
0.51 minutes. Timestamps are normalized to UTC by the consumer.

## Consumer processing guarantees

1. Validate the job payload and timestamps.
2. Ensure the unique MongoDB index on `queue_entry_id` exists before
   consuming jobs.
3. Persist a valid event to MongoDB before returning success.
4. Treat an identical duplicate as successful without inserting another
   event.
5. Permanently reject malformed events and conflicting duplicates.
6. Propagate transient database failures so BullMQ can retry the job.

`queue_entry_id` must be globally unique across restaurants for this
idempotency contract. If it is only unique within a restaurant, the
index and contract must instead use a compound key.

## Producer and consumer responsibilities

The producer owns job creation, attempts, backoff, and job retention.
The Python consumer owns validation, persistence, idempotency, and
classification of permanent versus retryable failures.

The consumer cannot enforce producer retry options after a job has
already been created.

## Integration validation before production

- Verify Node BullMQ v6 and Python bullmq==3.3.4 interoperability.
- Confirm the exact job name used by the producer.
- Confirm Redis authentication and TLS settings, if applicable.
- Verify retry behavior after a transient MongoDB failure.
- Verify malformed and conflicting jobs become failed without retrying.
- Verify failed-job retention and replay procedures.
- Confirm the target MongoDB database and index.
