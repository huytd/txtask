use std::{
    collections::HashMap,
    fs::{self, File},
    io::{stdin, stdout, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use text_splitter::TextSplitter;
use tokenizers::tokenizer::Tokenizer;
use tokio_stream::StreamExt;

#[derive(Serialize, Clone)]
enum Role {
    User,
    Assistant,
}

impl Role {
    fn as_str(&self) -> &str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

#[derive(Serialize, Clone)]
struct ChatMessage {
    role: Role,
    content: String,
}

type Embedding = Vec<f64>;

fn cosine_similarity(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len()); // Ensure both vectors have the same length

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let magnitude_a = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let magnitude_b = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

    dot_product / (magnitude_a * magnitude_b)
}

#[derive(Debug)]
struct ErrorMessage(String);

impl From<reqwest::Error> for ErrorMessage {
    fn from(value: reqwest::Error) -> Self {
        Self(format!("{value:?}"))
    }
}

impl From<std::io::Error> for ErrorMessage {
    fn from(value: std::io::Error) -> Self {
        Self(format!("{value:?}"))
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaEmbeddingsResponse {
    embedding: Embedding,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OllamaChatResponse {
    pub model: String,
    #[serde(rename = "created_at")]
    pub created_at: String,
    pub message: Message,
    pub done: bool,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub role: String,
    pub content: String,
}

const OLLAMA_API_BASE: &str = "http://localhost:11434/api";

struct OllamaClient {
    client: reqwest::Client,
}

impl OllamaClient {
    fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    async fn get_embeddings(&self, input: &str) -> Result<Embedding, ErrorMessage> {
        let resp: OllamaEmbeddingsResponse = self
            .client
            .post(format!("{OLLAMA_API_BASE}/embeddings"))
            .json(&serde_json::json!({
                "model": "mistral",
                "prompt": input.to_lowercase()
            }))
            .send()
            .await?
            .json()
            .await?;
        Ok(resp.embedding)
    }

    async fn get_answer(
        &self,
        input: &str,
        context: &[SimilarDocument],
        previous_messages: &[ChatMessage],
    ) -> Result<String, ErrorMessage> {
        let prompt = format!(
            "Based on the following context and this conversation, answer my next question.\n\nContext: {provided_context}\n\nQuestion: {input}",
            provided_context = context.iter().map(|doc| format!("File: {}\n```\n{}\n```", doc.source, doc.content)).collect::<Vec<String>>().join("\n\n")
        );
        let messages = [previous_messages, &[ChatMessage {
            role: Role::User,
            content: prompt,
        }]].concat();
        let resp = self
            .client
            .post(format!("{OLLAMA_API_BASE}/chat"))
            .json(&serde_json::json!({
                "model": "mistral",
                "stream": true,
                "messages": messages
            }))
            .send()
            .await?;

        print!("\x1b[1mAnswer:\x1b[0m");
        let mut body = resp.bytes_stream();
        let mut full_answer = String::new();
        while let Some(chunk) = body.next().await {
            let chunk = chunk.unwrap();
            let response: Result<OllamaChatResponse, serde_json::Error> =
                serde_json::from_reader(&*chunk);
            match response {
                Ok(data) => {
                    print!("{}", data.message.content);
                    full_answer.push_str(&data.message.content);
                    _ = stdout().flush();
                }
                _ => {}
            }
        }
        println!("");
        Ok(full_answer)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct EmbeddedDocument {
    source: String,
    embedding: Embedding,
}

#[derive(Debug)]
struct SimilarDocument {
    source: String,
    content: String,
    score: f64,
}

struct Database<'a> {
    data: HashMap<String, EmbeddedDocument>,
    api_client: &'a OllamaClient,
}

impl<'a> Database<'a> {
    fn new(api_client: &'a OllamaClient) -> Self {
        Self {
            data: HashMap::new(),
            api_client,
        }
    }

    fn save(&self) {
        let data = json!({ "data": self.data });
        let mut file = File::create("database.json").unwrap();
        file.write_all(data.to_string().as_bytes()).unwrap();
    }

    fn load(&mut self) {
        #[derive(Deserialize)]
        struct DBFile {
            data: HashMap<String, EmbeddedDocument>,
        }

        let content = fs::read_to_string("database.json").unwrap();
        let parsed: DBFile = serde_json::from_str(&content).unwrap();
        let mut filtered = HashMap::new();
        for key in parsed.data.keys() {
            filtered.insert(key.to_owned(), parsed.data.get(key).unwrap().to_owned());
        }
        self.data = filtered;
    }

    async fn add_document(&mut self, input: &str, source: &str) -> Result<(), ErrorMessage> {
        if !self.data.contains_key(input) {
            let embedding = self.api_client.get_embeddings(input).await?;
            self.data.insert(
                input.to_string(),
                EmbeddedDocument {
                    source: source.to_string(),
                    embedding,
                },
            );
        }
        Ok(())
    }

    async fn similarity_search(&self, input: &str) -> Result<Vec<SimilarDocument>, ErrorMessage> {
        let input = input.trim();
        let query_embedding = self.api_client.get_embeddings(&format!("{input}")).await?;
        let mut similarities: Vec<SimilarDocument> = vec![];
        for (content, doc) in &self.data {
            let score = cosine_similarity(&query_embedding, &doc.embedding);
            similarities.push(SimilarDocument {
                source: doc.source.to_owned(),
                content: content.to_owned(),
                score: score,
            });
        }
        similarities.sort_by(|a, b| b.score.total_cmp(&a.score));
        let avg_score = similarities.iter().fold(0., |score, doc| score + doc.score)
            / similarities.len() as f64;
        Ok(similarities
            .into_iter()
            .filter(|doc| doc.score >= avg_score)
            .collect())
    }
}

#[tokio::main]
async fn main() -> Result<(), ErrorMessage> {
    let api_client = OllamaClient::new();
    let mut database = Database::new(&api_client);

    if fs::metadata("database.json").is_ok() {
        database.load();
    } else {
        println!("Initialing database...");
        let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();
        let splitter = TextSplitter::new(tokenizer).with_trim_chunks(true);

        for entry in fs::read_dir("./data")? {
            let entry = entry?;
            let file_path = entry.path();
            let is_markdown = file_path.to_str().unwrap().ends_with("md");
            if !is_markdown {
                continue;
            }
            println!("Processing {file_path:?}...");
            let file_content = fs::read_to_string(&file_path)?;
            if !file_content.is_empty() {
                let mut i = 0;
                let chunks = splitter.chunks(&file_content, 100);
                for chunk in chunks {
                    i += 1;
                    println!("  Processing chunk {}", i);
                    let file_name = file_path.to_str().unwrap();
                    database.add_document(&chunk, file_name).await?;
                }
            }
        }
        database.save();
    }

    println!("Database initialized!\nNow you can ask me anything!\n\n");

    let mut is_running = true;
    let mut previous_messages: Vec<ChatMessage> = vec![];

    while is_running {
        print!("\x1b[1mQuestion:\x1b[0m ");
        _ = stdout().flush();

        let mut buf = String::new();
        stdin().read_line(&mut buf)?;

        let question = buf.trim();

        if question.eq("exit") {
            is_running = false;
        }

        println!("> Looking for answer...");

        if let Ok(matching) = database.similarity_search(question).await {
            println!("> Found {} records.", matching.len());
            let answer = api_client.get_answer(question, &matching, &previous_messages).await?;
            previous_messages.push(ChatMessage {
                role: Role::User,
                content: question.to_string(),
            });
            previous_messages.push(ChatMessage {
                role: Role::Assistant,
                content: answer,
            });
        }
    }

    println!("Bye!");
    Ok(())
}
