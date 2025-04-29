import { NextRequest, NextResponse } from "next/server";
import { ChatOpenAI } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { OpenAIEmbeddings } from "@langchain/openai";
import { getPineconeClient } from "@/app/lib/pinecone";

const pineconeIndexName = process.env.PINECONE_INDEX || "default-index-name";

async function queryPineconeWithText(query: string) {
  const pineconeClient = getPineconeClient();
  const index = pineconeClient.Index(pineconeIndexName);

  const apiKey = process.env.NEXT_PUBLIC_STUFF;
  if (!apiKey) throw new Error("Missing OpenAI API key");

  const embeddings = new OpenAIEmbeddings({ openAIApiKey: apiKey });

  const vectorStore = await PineconeStore.fromExistingIndex(embeddings, {
    pineconeIndex: index,
  });

  const results = await vectorStore.similaritySearch(query, 5);
  return results;
}

function createChatModel() {
  const apiKey = process.env.NEXT_PUBLIC_STUFF;
  if (!apiKey) throw new Error("Missing OpenAI API key");

  return new ChatOpenAI({
    modelName: "gpt-4",
    temperature: 0.7,
    openAIApiKey: apiKey,
  });
}

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();
    const chatModel = createChatModel();

    // 1. First try searching Pinecone
    const pineconeResults = await queryPineconeWithText(prompt);

    if (pineconeResults.length > 0) {
      // 🧠 Summarize the top 5 Pinecone results into a coherent response
      const combinedContent = pineconeResults
        .map((doc, i) => `Result ${i + 1}:\n${doc.pageContent}`)
        .join("\n\n");

      const summarizationPrompt = `
You are a GeneScope assistant. Based on the following research findings, answer the user's query concisely, professionally, and based only on the research.

User's Question:
"${prompt}"

Research Findings:
${combinedContent}
      `;

      const response = await chatModel.invoke([
        { role: "user", content: summarizationPrompt },
      ]);

      return NextResponse.json({ result: response.content });
    }

    // 2. If no good Pinecone results, just answer normally
    const fallbackResponse = await chatModel.invoke([
      { role: "user", content: prompt },
    ]);

    return NextResponse.json({ result: fallbackResponse.content });
  } catch (error) {
    console.error("LangChain error:", error);
    return NextResponse.json(
      {
        error: `Failed to get response: ${
          error instanceof Error ? error.message : String(error)
        }`,
      },
      { status: 500 }
    );
  }
}
