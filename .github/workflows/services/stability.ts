import type { AspectRatio } from '../types';

const STABILITY_API_KEY = process.env.STABILITY_API_KEY;
const API_HOST = 'https://api.stability.ai';
const ENGINE_ID = 'stable-image-generate-sd3';

// Helper to convert blob to base64
function blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            // result contains the data as a data URL, need to strip the prefix
            const base64String = (reader.result as string).split(',')[1];
            resolve(base64String);
        };
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

export async function generateImageWithStability(prompt: string, aspectRatio: AspectRatio): Promise<{ base64: string | null; error: string | null; }> {
    if (!STABILITY_API_KEY) {
        return { base64: null, error: "Stability AI API key is not configured." };
    }

    const formData = new FormData();
    formData.append('prompt', prompt);
    formData.append('aspect_ratio', aspectRatio);
    formData.append('output_format', 'jpeg');

    try {
        const response = await fetch(
            `${API_HOST}/v2beta/stable-image/generate/${ENGINE_ID}`,
            {
                method: 'POST',
                headers: {
                    Authorization: `Bearer ${STABILITY_API_KEY}`,
                    Accept: 'image/*',
                },
                body: formData,
            }
        );

        if (!response.ok) {
            const errorBody = await response.json();
            const errorMessage = errorBody.errors ? errorBody.errors[0] : 'Unknown Stability AI error';
            console.error('Stability AI Error:', errorMessage);
            throw new Error(errorMessage);
        }

        const imageBlob = await response.blob();
        const base64 = await blobToBase64(imageBlob);
        
        return { base64, error: null };

    } catch (err) {
        console.error('Stability AI request failed:', err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred with Stability AI.';
        return { base64: null, error: `Stability AI Error: ${errorMessage}` };
    }
}