import { NextRequest, NextResponse } from "next/server";
import { ChatOpenAI } from "@langchain/openai";
import { initializeAgentExecutorWithOptions } from "langchain/agents";
import { DynamicStructuredTool } from "langchain/tools";
import { z } from "zod";
//AAAAAAAAAAAAAAAaaa
//TODO -> fix prompt implement vector embedding tool
//graph is functional but struggling on generating input right after graph
const chatModel = new ChatOpenAI({
  modelName: "gpt-4",
  temperature: 0.7,
  openAIApiKey: process.env.NEXT_PUBLIC_STUFF,
});
interface ToolStep {
  action: { tool: string };
  observation: { content: string };
}
const generateChartTool = new DynamicStructuredTool({
  name: "generate_chart",

  description:
    "📊 MANDATORY: Use this tool to generate any bar, line, or pie chart for Chart.js. " +
    "DO NOT return code snippets or markdown descriptions. Only use this tool to return the chart data object. " +
    "Provide the type (bar, line, pie), title, labels, and data.",
  schema: z.object({
    type: z.enum(["bar", "line", "pie"]),
    title: z.string(),
    labels: z.array(z.string()),
    data: z.array(z.number()),
  }),
  func: async ({ type, title, labels, data }) => {
    return {
      name: "generate_chart",
      content: JSON.stringify({
        type,
        data: {
          labels,
          datasets: [
            {
              label: title,
              data,
              backgroundColor: [
                "rgba(255, 99, 132, 0.5)",
                "rgba(54, 162, 235, 0.5)",
                "rgba(255, 206, 86, 0.5)",
                "rgba(75, 192, 192, 0.5)",
                "rgba(153, 102, 255, 0.5)",
              ],
              borderColor: [
                "rgba(255, 99, 132, 1)",
                "rgba(54, 162, 235, 1)",
                "rgba(255, 206, 86, 1)",
                "rgba(75, 192, 192, 1)",
                "rgba(153, 102, 255, 1)",
              ],
              borderWidth: 1,
            },
          ],
        },
      }),
    };
  },
});

const isChartPrompt = async (input: string): Promise<boolean> => {
  const detectionPrompt = `
Determine if the following user prompt is asking to generate a chart.
Respond with only "yes" or "no".

Prompt: "${input}"
`;

  const response = await chatModel.invoke([
    { role: "user", content: detectionPrompt },
  ]);
  return (
    typeof response.content === "string" &&
    response.content.toLowerCase().includes("yes")
  );
};

export async function POST(req: NextRequest) {
  try {
    const { prompt } = await req.json();

    const shouldGenerateChart = await isChartPrompt(prompt);

    if (!shouldGenerateChart) {
      const response = await chatModel.invoke([
        { role: "user", content: prompt },
      ]);
      return NextResponse.json({ result: response.content });
    }

    const executor = await initializeAgentExecutorWithOptions(
      [generateChartTool],
      chatModel,
      {
        agentType: "openai-functions",
        verbose: true,
        returnIntermediateSteps: true,
      }
    );

    const wrappedPrompt = `
You are a data visualization assistant.
Only use the "generate_chart" tool to generate any chart.
Never return code blocks or markdown.
Only return the chart JSON using the tool.

User prompt: ${prompt}
`;

    const result = await executor.invoke({ input: wrappedPrompt });
    const toolStep = result?.intermediateSteps?.find(
      (step: ToolStep) => step?.action?.tool === "generate_chart"
    );

    if (toolStep?.observation?.content) {
      const parsedContent = JSON.parse(toolStep.observation.content);
      return NextResponse.json({ result: parsedContent });
    }

    return NextResponse.json({ result: result.output });
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
