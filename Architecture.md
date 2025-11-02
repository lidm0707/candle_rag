# Candle RAG System Architecture

## Overview

The Candle RAG system is a Rust-based text-to-vector transformation pipeline designed to convert text inputs into semantic embeddings for database storage and retrieval. Built on the Candle ML framework, this system provides efficient text embedding generation for Retrieval-Augmented Generation (RAG) applications.

## System Goals

1. **Text-to-Vector Transformation**: Convert arbitrary text into high-dimensional semantic vectors
2. **Database Integration**: Store and retrieve vectors efficiently for similarity search
3. **Performance**: Optimize for speed and memory efficiency using Candle's lightweight design
4. **Scalability**: Support batch processing and concurrent operations
5. **Extensibility**: Modular design supporting multiple embedding models

## Core Components

### 1. Embedding Engine (`src/embedding/`)

**Purpose**: Core text-to-vector transformation logic

**Key Modules**:
- `model_loader.rs` - Download and load pre-trained models from HuggingFace Hub
- `text_encoder.rs` - Handle text tokenization and encoding
- `embedding_generator.rs` - Generate embeddings from encoded text
- `model_registry.rs` - Manage multiple embedding models

**Supported Models**:
- BERT (base variants)
- BGE (BAAI/bge-large-en-v1.5, BAAI/bge-base-en-v1.5)
- Jina-BERT
- DistilBERT
- Custom transformer models

### 2. Vector Storage (`src/storage/`)

**Purpose**: Abstract vector database operations

**Key Modules**:
- `vector_store.rs` - Trait defining vector storage interface
- `sqlite_store.rs` - SQLite implementation with vector extensions
- `postgres_store.rs` - PostgreSQL implementation with pgvector
- `memory_store.rs` - In-memory storage for testing and caching
- `index_manager.rs` - Manage vector indexing for similarity search

**Storage Schema**:
```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    vector BLOB NOT NULL,  -- Serialized f32 array
    model_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE TABLE embedding_models (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    config JSON
);
```

### 3. API Layer (`src/api/`)

**Purpose**: External interface for the embedding system

**Key Modules**:
- `rest_api.rs` - HTTP REST endpoints
- `grpc_api.rs` - gRPC interface for high-performance scenarios
- `cli.rs` - Command-line interface for batch operations
- `websocket.rs` - Real-time embedding generation

**API Endpoints**:
```
POST /api/v1/embed
GET  /api/v1/similarity
POST /api/v1/batch-embed
GET  /api/v1/models
GET  /api/v1/health
```

### 4. Processing Pipeline (`src/pipeline/`)

**Purpose**: Coordinate end-to-end text processing workflows

**Key Modules**:
- `text_processor.rs` - Text preprocessing and chunking
- `batch_processor.rs` - Handle bulk embedding generation
- `similarity_search.rs` - Vector similarity operations
- `workflow_orchestrator.rs` - Coordinate complex processing workflows

### 5. Configuration (`src/config/`)

**Purpose**: System configuration and settings management

**Key Modules**:
- `model_config.rs` - Model-specific configurations
- `database_config.rs` - Database connection settings
- `performance_config.rs` - Performance tuning parameters
- `feature_flags.rs` - Feature toggles and experiments

## Data Flow Architecture

### Text-to-Vector Pipeline

```
Input Text
    ↓
Text Preprocessing
    ↓
Tokenization
    ↓
Model Inference (Candle)
    ↓
Embedding Tensor
    ↓
Vector Normalization
    ↓
Database Storage
```

### Similarity Search Flow

```
Query Text
    ↓
Embedding Generation
    ↓
Vector Index Lookup
    ↓
Similarity Calculation (Cosine/Manhattan)
    ↓
Ranked Results
```

## Performance Considerations

### 1. Model Loading Optimization
- **Model Caching**: Keep frequently used models in memory
- **Lazy Loading**: Load models on-demand with thread-safe initialization
- **Model Sharing**: Share model instances across concurrent requests

### 2. Batch Processing
- **Dynamic Batching**: Group requests automatically for optimal throughput
- **Memory Management**: Efficient tensor allocation and deallocation
- **Parallel Processing**: Multi-threaded embedding generation

### 3. Vector Operations
- **SIMD Optimization**: Leverage CPU vector instructions
- **Approximate Search**: Use HNSW or other ANN algorithms for large datasets
- **Caching**: Cache recent similarity search results

## Security Architecture

### 1. Input Validation
- Text length limits and sanitization
- Model parameter validation
- Rate limiting per client/API key

### 2. Data Protection
- Encrypted storage for sensitive text
- Access control for embedding models
- Audit logging for all operations

### 3. Model Security
- Model integrity verification (hash checking)
- Secure model distribution from HuggingFace Hub
- Sandboxed model execution environments

## Deployment Architecture

### 1. Single Node Deployment
```
┌─────────────────┐
│   Application   │
├─────────────────┤
│  SQLite Store   │
├─────────────────┤
│  Model Cache    │
└─────────────────┘
```

### 2. Distributed Deployment
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   API GW    │────│  Load Bal   │────│   App Pods  │
└─────────────┘    └─────────────┘    └─────────────┘
                                            │
                       ┌────────────────────┼────────────────────┐
                       │                    │                    │
                ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
                │Vector Store │      │ Model Cache │      │  Monitoring │
                └─────────────┘      └─────────────┘      └─────────────┘
```

## Technology Stack

### Core Framework
- **Candle**: Lightweight ML framework for Rust
- **Tokio**: Async runtime for concurrent operations
- **Serde**: Serialization/deserialization

### Database Options
- **SQLite**: For small to medium deployments
- **PostgreSQL + pgvector**: For production workloads
- **Redis**: For caching and real-time operations

### Model Sources
- **HuggingFace Hub**: Primary source for pre-trained models
- **Local Models**: Custom fine-tuned models
- **Model Registry**: Version control for model deployments

## Development Roadmap

### Phase 1: Core Functionality
- [ ] Basic text-to-vector transformation
- [ ] Model loading and management
- [ ] Simple in-memory storage
- [ ] CLI interface

### Phase 2: Database Integration
- [ ] SQLite vector storage implementation
- [ ] Similarity search functionality
- [ ] Batch processing capabilities
- [ ] REST API development

### Phase 3: Production Features
- [ ] PostgreSQL/pgvector support
- [ ] Advanced indexing (HNSW)
- [ ] Performance optimizations
- [ ] Monitoring and observability

### Phase 4: Advanced Features
- [ ] Multi-modal embeddings (text + images)
- [ ] Fine-tuning pipeline integration
- [ ] Advanced RAG workflows
- [ ] Distributed deployment support

## Example Usage

### Basic Text Embedding
```rust
use candle_rag::embedding::{EmbeddingGenerator, ModelConfig};

let generator = EmbeddingGenerator::new(ModelConfig::bge_base())?;
let vector = generator.embed("hello world")?;
println!("Generated vector: {:?}", vector);
```

### Database Storage
```rust
use candle_rag::storage::{VectorStore, EmbeddingRecord};

let store = VectorStore::sqlite("embeddings.db")?;
let record = EmbeddingRecord {
    text: "hello world".to_string(),
    vector: embedding,
    model_id: "bge-base-en-v1.5".to_string(),
    metadata: None,
};
store.insert(record).await?;
```

### Similarity Search
```rust
let query_vector = generator.embed("greeting message")?;
let similar = store.search_similar(query_vector, 10).await?;
for result in similar {
    println!("{} (similarity: {:.4})", result.text, result.score);
}
```

This architecture provides a solid foundation for building a production-ready RAG system using Candle, with clear separation of concerns, scalability considerations, and extensibility for future enhancements.