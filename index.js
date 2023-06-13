import * as dotenv from "dotenv";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { PineconeClient } from "@pinecone-database/pinecone";
import { PineconeStore } from "langchain/vectorstores/pinecone";

dotenv.config();

// Create docs with a loader
const loader = new TextLoader("sample_texts/HistoryOfUSA.txt");
const docs = await loader.load();

const client = new PineconeClient();
await client.init({
  apiKey: process.env.PINECONE_API_KEY,
  environment: process.env.PINECONE_ENVIRONMENT,
});
const pineconeIndex = client.Index(process.env.PINECONE_INDEX);
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

await PineconeStore.fromDocuments(docs, embeddings, {
  pineconeIndex,
});

const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
  pineconeIndex,
});

/* Search the vector DB independently with meta filters */
const results = await vectorStore.similaritySearch("Science", 1);
console.log(results);
