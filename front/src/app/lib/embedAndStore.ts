import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/community/vectorstores/pinecone";
import { getPineconeClient } from "./pinecone.js";
//key sotred
//TODO -> make it more efficient
const pineconeIndexName = process.env.PINECONE_INDEX || "default-index-name";

export async function storePdfsInPinecone() {
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.NEXT_PUBLIC_STUFF,
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
      pineconeIndex: getPineconeClient().Index(pineconeIndexName),
    });

    console.log("PDF embeddings stored in Pinecone.");
  } catch (error) {
    console.error("Error storing PDF embeddings in Pinecone:", error);
  }
}
