
import { GoogleGenAI, Type } from '@google/genai';
import type { AspectRatio } from '../types';

if (!process.env.API_KEY) {
    throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const PROMPT_GENERATION_MODEL = 'gemini-2.5-flash';
const IMAGE_GENERATION_MODEL = 'imagen-3.0-generate-002';

const promptSchema = {
  type: Type.OBJECT,
  properties: {
    prompts: {
      type: Type.ARRAY,
      items: {
        type: Type.STRING,
        description: 'A single, descriptive prompt for generating an image based on a scene from the script.',
      },
    },
  },
  required: ['prompts'],
};

export async function generatePrompts(script: string, style: string, niche: string): Promise<{ prompts: string[], requestPrompt: string }> {
    const requestPrompt = `System Instruction: You are an expert script analyst and creative director. Your job is to read a script, break it down into key visual moments, and generate safe, detailed prompts for a text-to-image AI.

User Request:
I have a script that needs to be visualized. Please generate a series of image prompts based on it.

**Context & Style:**
${niche ? `- **Topic/Niche:** ${niche}\n` : ''}- **Visual Style:** ${style}

**CRITICAL INSTRUCTIONS:**
1.  **Analyze and Breakdown:** Read the script and divide it into logical scenes or distinct visual moments.
2.  **Generate Prompts:** For each scene, create ONE image prompt. Each prompt MUST be detailed and strictly adhere to the provided **Visual Style** and incorporate the **Topic/Niche** (if provided).
3.  **IMPORTANT SAFETY RULE:** You MUST generate prompts that are safe and appropriate for a general audience. Do not describe or imply violence, explicit situations, or sensitive interactions, especially those involving minors, even if historically accurate. If the script contains such themes, you MUST represent them abstractly or symbolically. Focus on setting, atmosphere, and emotion. For example, to show tension, describe "long, distorted shadows in a dimly lit room" instead of a direct confrontation. Failure to follow this rule will result in an invalid response.
4.  **Output Format:** Your entire response MUST be a valid JSON object with a single key "prompts", which is an array of strings. Each string in the array is a single image prompt. Do not add any commentary, explanations, or markdown formatting around the JSON.

**Example Response:**
{
  "prompts": [
    "A lone astronaut stands on a desolate red planet, facing a swirling dust storm under a dim sun. Cinematic lighting casts long, dramatic shadows. The style is reminiscent of Denis Villeneuve's 'Dune'.",
    "Extreme close-up on the astronaut's cracked helmet visor. The glass reflects a tiny, distant blue Earth, a stark contrast to the harsh alien landscape. The image is hyperrealistic, with visible dust particles floating in the foreground."
  ]
}

**SCRIPT TO ANALYZE:**
---
${script}
---
`;

    try {
        const response = await ai.models.generateContent({
            model: PROMPT_GENERATION_MODEL,
            contents: requestPrompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: promptSchema,
            },
        });
        
        const jsonText = response.text;
        
        if (!jsonText) {
            console.error("AI response text is empty or undefined. Full response:", response);
            const finishReason = response.candidates?.[0]?.finishReason;
            let errorMessage = "The AI returned an empty response. This can happen if the script is too short, vague, or contains content that goes against the safety policy.";

            if (finishReason === 'SAFETY') {
                errorMessage = "The script or style contains content that violates safety policies. Please revise your input and try again.";
            } else if (finishReason === 'RECITATION') {
                errorMessage = "The response was blocked due to potential recitation issues. Try rephrasing your script.";
            } else if (finishReason && finishReason !== 'STOP') {
                errorMessage = `Prompt generation stopped unexpectedly. Reason: ${finishReason}. Please check your script content.`;
            }
            throw new Error(errorMessage);
        }

        try {
            const result = JSON.parse(jsonText);
            if (result && Array.isArray(result.prompts)) {
                return { prompts: result.prompts, requestPrompt };
            } else {
                throw new Error('The AI returned a response with an invalid structure. Please try again.');
            }
        } catch (parseError) {
             console.error("Failed to parse AI response as JSON:", jsonText, parseError);
             throw new Error("The AI returned a response that was not valid JSON. This may be a temporary issue, please try again.");
        }

    } catch (error) {
        console.error("Error during prompt generation:", error);
        // Re-throw specific, user-friendly errors, otherwise provide a generic one.
        if (error instanceof Error && (
            error.message.startsWith("The AI returned") ||
            error.message.startsWith("The script or style") ||
            error.message.startsWith("The response was blocked") ||
            error.message.startsWith("Prompt generation stopped")
        )) {
            throw error;
        }
        throw new Error("Failed to generate prompts from the script due to an unexpected AI service error.");
    }
}

export async function analyzeImageStyle(imageData: { base64: string, mimeType: string }): Promise<string> {
    const imagePart = {
        inlineData: {
            mimeType: imageData.mimeType,
            data: imageData.base64,
        },
    };
    const textPart = {
        text: "Analyze the artistic style of this image. Describe the style in a concise, comma-separated list of keywords and phrases suitable for a text-to-image AI. Focus on elements like lighting, color palette, composition, medium (e.g., photograph, oil painting), and overall mood. Do not use full sentences. Example: cinematic, dramatic lighting, high contrast, muted color palette, photorealistic, shallow depth of field, moody atmosphere.",
    };

    try {
        const response = await ai.models.generateContent({
            model: PROMPT_GENERATION_MODEL,
            contents: { parts: [imagePart, textPart] },
        });
        return response.text.trim();
    } catch (error) {
        console.error("Error analyzing image style:", error);
        throw new Error("Failed to analyze the reference image style.");
    }
}


export async function generateImage(prompt: string, aspectRatio: AspectRatio): Promise<{ base64: string | null; error: string | null; }> {
    try {
        const response = await ai.models.generateImages({
            model: IMAGE_GENERATION_MODEL,
            prompt: prompt,
            config: {
                numberOfImages: 1,
                outputMimeType: 'image/jpeg',
                aspectRatio: aspectRatio,
            }
        });

        if (response.generatedImages && response.generatedImages.length > 0) {
            return { base64: response.generatedImages[0].image.imageBytes, error: null };
        } else {
            return { base64: null, error: 'The API did not return an image.' };
        }
    } catch(err) {
        console.error(`Image generation failed for prompt: "${prompt}"`, err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
        if (errorMessage.includes("sensitive words") || errorMessage.includes("Responsible AI practices") || errorMessage.includes("prompt contains sensitive words")) {
             return { base64: null, error: 'This prompt was blocked for safety reasons. Please try rephrasing it.' };
        }
        return { base64: null, error: 'Image generation failed.' };
    }
}