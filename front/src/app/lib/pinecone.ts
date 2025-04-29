//src/app/lib/embedAndStore.ts
import { Pinecone } from "@pinecone-database/pinecone";

let client: Pinecone | null = null;

export function getPineconeClient(): Pinecone {
  if (!client) {
    const apiKey = process.env.PINECONE_API_KEY;
    if (!apiKey) {
      throw new Error("PINECONE_API_KEY is not set.");
    }

    client = new Pinecone({ apiKey });
  }

  return client;
}
