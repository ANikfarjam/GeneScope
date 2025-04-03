import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { pineconeClient } from "./pinecone.js";

export async function storePdfsInPinecone() {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey:
      "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",
  });
  const stagingDocs = await new PDFLoader("data/BRCAStaging.pdf").load();
  const hmmDocs = await new PDFLoader(
    "data/cancerClassificationHMM.pdf"
  ).load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });

  const allDocs = [
    ...(await splitter.splitDocuments(stagingDocs)),
    ...(await splitter.splitDocuments(hmmDocs)),
  ];

  try {
    await PineconeStore.fromDocuments(allDocs, embeddings, {
      pineconeIndex: pineconeClient.Index("gene-scope"),
    });

    console.log("PDF embeddings stored in Pinecone.");
  } catch (error) {
    console.error("Error storing PDF embeddings in Pinecone:", error);
  }
}
