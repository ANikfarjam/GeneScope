import { Pinecone } from "@pinecone-database/pinecone";
//where key stored
//TODO -> security
const pineconeClient = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY || "",
});

export { pineconeClient };
