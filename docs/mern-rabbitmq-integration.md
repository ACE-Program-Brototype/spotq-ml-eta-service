# MERN–Python ETA RabbitMQ Integration Contract (SCRUM-1257)

This document specifies the RabbitMQ integration contract between the MERN backend (event producer) and the Python ETA microservice (event consumer), replacing legacy BullMQ.

---

## 1. Architecture Overview

When a customer completes their queue wait or is seated at a restaurant, the MERN backend publishes an event to RabbitMQ. The Python ETA microservice asynchronously consumes this event, validates and normalizes the payload, auto-calculates elapsed wait time if omitted, and durably persists the record into MongoDB Atlas (`queue_events` collection). These records serve as historical ground truth for continuous ML model training and feature pre-warming.

```text
┌──────────────────────┐
│     MERN Backend     │
│   (Event Producer)   │
└──────────┬───────────┘
           │ publishes AMQP 0-9-1 message
           │ Routing Key: queue.event.completed
           ▼
┌────────────────────────────────────────────────────────┐
│                   RabbitMQ Broker                      │
│                                                        │
│  Topic Exchange: `spotq.events`                        │
│         │                                              │
│         ├─ Binding (`queue.event.completed`)           │
│         │                                              │
│         ▼                                              │
│  Durable Queue: `ml-eta-queue`                         │
│  [x-dead-letter-exchange: ""]                          │
│  [x-dead-letter-routing-key: "ml-eta-queue.dlq"]       │
│                                                        │
│  Dead-Letter Queue: `ml-eta-queue.dlq`                 │
└──────────┬───────────────────────────────▲─────────────┘
           │ consume (prefetch: 10)         │
           │                                │ Reject (requeue=False)
           ▼                                │ on poison pill / bad schema
┌──────────────────────────────────────────┴─────────────┐
│             Python ETA Microservice                    │
│             `RabbitMQConsumer` worker                  │
│                                                        │
│  1. Decode JSON                                        │
│  2. Validate ISO 8601 Timestamps & Schema              │
│  3. Auto-calculate wait time if omitted                │
│  4. Idempotency Check & Persist                        │
└──────────┬─────────────────────────────────────────────┘
           │
           │ ACK on success
           ▼
┌──────────────────────┐
│    MongoDB Atlas     │
│ Collection:          │
│ `queue_events`       │
└──────────────────────┘
```

---

## 2. RabbitMQ Topology & AMQP Configuration

| Component | Identifier | Type / Details |
|---|---|---|
| **Exchange** | `spotq.events` | Topic exchange, `durable=True` |
| **Main Queue** | `ml-eta-queue` | Queue, `durable=True`, configured with DLX |
| **Dead-Letter Queue** | `ml-eta-queue.dlq` | Queue, `durable=True`, holds poison pills and unparseable events |
| **Routing Key** | `queue.event.completed` | Standard routing key for queue completion events |
| **Dead-Letter Exchange** | `""` (default exchange) | Direct delivery to `ml-eta-queue.dlq` |
| **Prefetch Count** | `10` (configurable) | QoS prefetch limit per consumer instance |
| **Delivery Mode** | `2` (Persistent) | Producer should publish persistent messages |

---

## 3. Event Payload Contract

Messages published by the MERN backend must be UTF-8 encoded JSON payloads conforming to the schema below:

### Schema Fields

| Field | Type | Required | Description & Constraints |
|---|---|---|---|
| `queue_entry_id` | `string` | **Yes** | Globally unique identifier for the queue entry (max 128 characters). Used for idempotency. |
| `restaurant_id` | `string` | **Yes** | Unique identifier for the restaurant (max 128 characters). |
| `party_size` | `integer` | **Yes** | Number of people in the queue party (1 to 100). |
| `joined_at` | `string (ISO 8601)` | **Yes** | Timezone-aware ISO 8601 timestamp when user joined queue (e.g. `2026-10-18T12:00:00Z`). |
| `completed_at` | `string (ISO 8601)` | **Yes** | Timezone-aware ISO 8601 timestamp when wait concluded. Must be strictly after `joined_at`. |
| `actual_wait_time_minutes` | `number \| null` | *Optional* | Elapsed wait duration in minutes. If omitted or `null`, **automatically calculated by the Python service** as `(completed_at - joined_at).total_seconds() / 60.0`. If provided, must match elapsed time within ±0.51 minutes tolerance. |
| `total_people_ahead_at_join` | `integer` | **Yes** | Number of people ahead in queue at time of entry (0 to 1,000). |

### Example Payloads

#### Standard Payload (with wait time):
```json
{
  "queue_entry_id": "qe_9876543210abcdef",
  "restaurant_id": "rest_central_bistro_01",
  "party_size": 4,
  "joined_at": "2026-10-18T19:00:00Z",
  "completed_at": "2026-10-18T19:24:30Z",
  "actual_wait_time_minutes": 24.5,
  "total_people_ahead_at_join": 8
}
```

#### Minimal Payload (omitting wait time; auto-calculated by Python):
```json
{
  "queue_entry_id": "qe_1234567890abcdef",
  "restaurant_id": "rest_central_bistro_01",
  "party_size": 2,
  "joined_at": "2026-10-18T19:00:00Z",
  "completed_at": "2026-10-18T19:15:00Z",
  "total_people_ahead_at_join": 3
}
```

---

## 4. Message Reliability & Delivery Semantics

The consumer enforces strict at-least-once delivery with reliable acknowledgement:

1. **Explicit Acknowledgement (`ACK`)**:
   - The message is acknowledged (`message.ack()`) **only after** the event is successfully inserted into MongoDB Atlas or confirmed as an identical duplicate.
2. **Dead-Letter Routing (`Reject without requeue`)**:
   - Poison pills (malformed JSON, invalid schema, negative wait time, reversed timestamps, or conflicting duplicates) are permanently rejected with `message.reject(requeue=False)`.
   - RabbitMQ automatically routes these rejected messages into `ml-eta-queue.dlq` for operator investigation without blocking the consumer pipeline or causing infinite retry loops.
3. **Transient Failure Recovery (`NACK with requeue`)**:
   - Database disconnections or transient MongoDB network timeouts result in `message.nack(requeue=True)`. The broker requeues the message for a subsequent delivery attempt.
4. **Broker Connection Recovery**:
   - The consumer implements an automated reconnection loop with exponential backoff (starting at 2s, capped at 60s). When the RabbitMQ broker recovers, the consumer automatically re-establishes channels, redeclares topologies, and resumes message consumption.

---

## 5. Idempotency & Duplicate Protection

Network retries or broker redeliveries can cause duplicate message delivery:

- **Identical Duplicates**: If a message with an existing `queue_entry_id` arrives and all business fields match the existing record, the consumer logs an informational message and returns success (`ACK`). No duplicate record is inserted.
- **Conflicting Duplicates**: If a message arrives with an existing `queue_entry_id` but conflicting data (e.g., differing timestamps or restaurant IDs), the consumer rejects it to DLQ (`requeue=False`) to preserve historical integrity.

---

## 6. Environment Variables

Configure the following variables via `.env` or Infisical:

| Variable | Default | Purpose |
|---|---|---|
| `RABBITMQ_URL` | *None* | Complete AMQP connection URI (e.g. `amqp://user:pass@host:5672/`) |
| `RABBITMQ_HOST` | `localhost` | RabbitMQ broker host |
| `RABBITMQ_PORT` | `5672` | RabbitMQ broker port |
| `RABBITMQ_USER` | `guest` | RabbitMQ username |
| `RABBITMQ_PASSWORD` | `guest` | RabbitMQ password |
| `RABBITMQ_VHOST` | `/` | Virtual host |
| `RABBITMQ_QUEUE_NAME` | `ml-eta-queue` | Ingestion queue name |
| `RABBITMQ_EXCHANGE_NAME` | `spotq.events` | Exchange name |
| `RABBITMQ_ROUTING_KEY` | `queue.event.completed` | Binding routing key |
| `RABBITMQ_PREFETCH_COUNT` | `10` | Consumer prefetch QoS |
| `RABBITMQ_RECONNECT_INTERVAL_SECONDS` | `5.0` | Base reconnect interval |

---

## 7. Observability & Prometheus Metrics

The consumer exports real-time Prometheus metrics at `GET /metrics`:

| Metric Name | Type | Description |
|---|---|---|
| `rabbitmq_consumer_healthy` | Gauge | `1` when RabbitMQ consumer is actively connected, `0` otherwise |
| `rabbitmq_messages_received_total` | Counter | Total messages dequeued from RabbitMQ |
| `rabbitmq_messages_processed_total` | Counter | Total messages successfully processed and persisted |
| `rabbitmq_messages_failed_total` | Counter | Total transient message failures (NACKed with requeue) |
| `rabbitmq_messages_rejected_total` | Counter | Total unrecoverable/malformed messages rejected to DLQ |
| `rabbitmq_message_processing_duration_seconds` | Histogram | Latency histogram for message processing and DB persistence |

---

## 8. Verification & CLI Tooling

Two helper scripts are provided in `scripts/`:

1. **Publish test events**:
   ```bash
   uv run python scripts/produce_rabbitmq_event.py --type valid
   uv run python scripts/produce_rabbitmq_event.py --type no-wait
   uv run python scripts/produce_rabbitmq_event.py --type malformed
   ```

2. **Run end-to-end integration test**:
   ```bash
   uv run python scripts/verify_rabbitmq_integration.py
   ```

