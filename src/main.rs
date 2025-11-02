use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use hf_hub::{Repo, RepoType, api::sync::Api};
use libsql::{Builder, Connection};
use sqlite_vec::sqlite3_vec_init;
use std::{fs, path::PathBuf};
use tokenizers::Tokenizer;
use zerocopy::AsBytes;

async fn setup_vector_db() -> Result<Connection> {
    // Create local Turso database file
    let db_path = "db/turso.db";

    // Remove existing database file if it exists
    if std::path::Path::new(db_path).exists() {
        fs::remove_file(db_path)?;
    }

    let db = Builder::new_local(db_path).build().await?;
    let conn = db.connect()?;

    // Initialize sqlite-vec extension
    unsafe {
        libsql::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite3_vec_init as *const (),
        )));
    }

    // Create vector table
    conn.execute(
        "CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            embedding F32_BLOB(384)
        )",
        (),
    )
    .await?;

    // Create vector index
    conn.execute(
        "CREATE INDEX documents_idx ON documents (
            libsql_vector_idx(embedding, 'metric=cosine')
        )",
        (),
    )
    .await?;

    println!("✅ Vector database created at {}", db_path);
    Ok(conn)
}

async fn store_embedding(conn: &Connection, text: &str, embedding: &[f32]) -> Result<i64> {
    conn.execute(
        "INSERT INTO documents (text, embedding) VALUES (?1, ?2)",
        (text, embedding.as_bytes()),
    )
    .await?;

    // Get the last inserted row ID
    let mut rows = conn.query("SELECT last_insert_rowid()", ()).await?;
    let row = rows.next().await?.unwrap();
    let row_id = row.get::<i64>(0).unwrap();
    println!("✅ Stored embedding for '{}' with ID: {}", text, row_id);
    Ok(row_id)
}

async fn search_similar(
    conn: &Connection,
    query_embedding: &[f32],
    limit: i32,
) -> Result<Vec<(String, f64)>> {
    let mut rows = conn
        .query(
            "SELECT d.text, vector_distance_cos(d.embedding, vector(?1)) as distance
         FROM vector_top_k('documents_idx', vector(?1), ?2) as v
         JOIN documents d ON d.rowid = v.id
         ORDER BY distance",
            (query_embedding.as_bytes(), limit),
        )
        .await?;

    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        let text = row.get(0).unwrap_or_default();
        let distance = row.get::<f64>(1).unwrap_or_default();
        results.push((text, distance));
    }

    Ok(results)
}

async fn process_text(
    text: &str,
    model: &BertModel,
    tokenizer: &Tokenizer,
    device: &Device,
    _conn: &Connection,
) -> Result<Vec<f32>> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize: {}", e))?;

    println!("📝 Tokenized '{}':", text);
    println!("   Tokens: {:?}", encoding.get_tokens());
    println!("   IDs: {:?}", encoding.get_ids());

    // Get only the actual tokens (without padding)
    let tokens = encoding.get_tokens();
    let ids = encoding.get_ids();

    // Find the actual sequence length (exclude [PAD] tokens)
    let mut actual_length = 0;
    for token in tokens {
        if token == "[PAD]" {
            break;
        }
        actual_length += 1;
    }

    let trimmed_ids = &ids[..actual_length];
    let input_ids = Tensor::new(trimmed_ids, device)?.unsqueeze(0)?;
    let token_type_ids = Tensor::zeros(input_ids.shape(), DType::U32, device)?;

    // Create attention mask (1 for real tokens, 0 for padding)
    let attention_mask = Tensor::ones(input_ids.shape(), DType::U32, device)?;

    let output = model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?;
    println!("✅ Output shape: {:?}", output.shape());
    println!("🎯 Successfully processed: '{}'", text);
    println!("📊 Actual tokens: {:?}", &tokens[..actual_length]);
    println!("📊 Actual sequence length: {} tokens", actual_length);

    // Extract mean pooling of the last hidden state to get embedding
    // Shape: [batch_size, seq_len, hidden_dim] -> [hidden_dim]
    let last_hidden_state = &output;
    let (_batch_size, _seq_len, _hidden_dim) = last_hidden_state.dims3()?;

    // Simple mean pooling across sequence length
    let summed = last_hidden_state.sum_keepdim(1)?; // [batch_size, 1, hidden_dim]
    let embedding = summed.squeeze(1)?; // [batch_size, hidden_dim]
    let embedding = embedding.squeeze(0)?; // [hidden_dim]
    let embedding_vec = embedding.to_vec1()?;

    println!("🔢 Embedding dimension: {}", embedding_vec.len());
    println!();

    Ok(embedding_vec)
}

fn main() -> Result<()> {
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async_main())
}

async fn async_main() -> Result<()> {
    // 1) setup vector database
    let conn = setup_vector_db().await?;

    // 2) device
    let device = Device::Cpu;

    // 3) download via hf-hub (public model)
    let api = Api::new()?;
    let repo = api.repo(Repo::new(
        "sentence-transformers/all-MiniLM-L6-v2".to_string(),
        RepoType::Model,
    ));
    let config_remote = repo.get("config.json")?;
    let weights_remote = repo.get("model.safetensors")?;
    let tokenizer_remote = repo.get("tokenizer.json")?;

    // 4) save local
    let model_dir = PathBuf::from("models/all-MiniLM-L6-v2");
    fs::create_dir_all(&model_dir)?;
    let config_path = model_dir.join("config.json");
    let weights_path = model_dir.join("model.safetensors");
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !config_path.exists() {
        fs::copy(config_remote, &config_path)?;
    }
    if !weights_path.exists() {
        fs::copy(weights_remote, &weights_path)?;
    }
    if !tokenizer_path.exists() {
        fs::copy(tokenizer_remote, &tokenizer_path)?;
    }

    // 5) load config + model
    let config: Config = serde_json::from_slice(&fs::read(&config_path)?)?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path.clone()], DType::F32, &device)?
    };
    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // No prefix needed for sentence transformer model
    let model = BertModel::load(vb, &config)?;
    println!("✅ Model loaded");

    // 6) process and store text examples
    println!("🚀 Processing and storing text examples:\n");

    // Process "helloworld" and store embedding
    let helloworld_embedding =
        process_text("helloworld", &model, &tokenizer, &device, &conn).await?;
    store_embedding(&conn, "helloworld", &helloworld_embedding).await?;

    // Process "bye bot" and store embedding
    let bye_bot_embedding = process_text("bye bot", &model, &tokenizer, &device, &conn).await?;
    store_embedding(&conn, "bye bot", &bye_bot_embedding).await?;

    // Process additional examples
    let examples = vec![
        "Hello world!",
        "Good morning everyone",
        "This is a test sentence for BERT processing.",
        "Machine learning is fascinating",
        "Goodbye and farewell",
        "See you later",
        "Artificial intelligence",
        "Natural language processing",
        "092-222-2221",
    ];

    for example in examples {
        let embedding = process_text(example, &model, &tokenizer, &device, &conn).await?;
        store_embedding(&conn, example, &embedding).await?;
    }

    // 7) Search for similar documents
    println!("\n🔍 Searching for similar documents to 'helloworld':");
    let results = search_similar(&conn, &helloworld_embedding, 3).await?;
    for (text, distance) in results {
        println!("   {:.4} - {}", distance, text);
    }

    println!("\n🔍 Searching for similar documents to 'bye bot':");
    let results = search_similar(&conn, &bye_bot_embedding, 3).await?;
    for (text, distance) in results {
        println!("   {:.4} - {}", distance, text);
    }

    Ok(())
}
