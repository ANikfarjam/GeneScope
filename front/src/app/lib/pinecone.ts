import { Pinecone } from "@pinecone-database/pinecone";

const pineconeClient = new Pinecone({
  apiKey:
    "pcsk_6cNGsD_PH5j5fHSHxyyCA33fghRxC5TxBwGE9sQjHZboUdVFhzyQp8CmW7s3mLJHwHsKxW",
});

export { pineconeClient };
