import { NextRequest, NextResponse } from "next/server";
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0.7,
  openAIApiKey: process.env.NEXT_PUBLIC_STUFF, // from your .env
});

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();

    const response = await chatModel.invoke([
      { role: "user", content: prompt },
    ]);

    return NextResponse.json({ result: response.content });
  } catch (error) {
    console.error("LangChain error:", error);
    return NextResponse.json(
      { error: "Failed to get response" },
      { status: 500 }
    );
  }
}
