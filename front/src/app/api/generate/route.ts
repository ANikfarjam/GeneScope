import { NextRequest, NextResponse } from "next/server";
import { OpenAI } from "openai";

const openai = new OpenAI({
  apiKey: process.env.NEXT_PUBLIC_STUFF, // Ensure this is set in .env
});

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();
    const response = await openai.chat.completions.create({
      model: "gpt-4", 
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });

    return NextResponse.json({ result: response.choices[0].message?.content });
  } catch (error) {
    console.error("OpenAI API Error Please inser a key:", error);
    return NextResponse.json({ error: "Failed to fetch response" }, { status: 500 });
  }
}
