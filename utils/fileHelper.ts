import mammoth from 'mammoth';
import JSZip from 'jszip';
import saveAs from 'file-saver';
import type { GeneratedImage } from '../types';

export async function readTextFromFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = async (event) => {
      if (!event.target?.result) {
        return reject(new Error('Failed to read file.'));
      }

      if (file.name.endsWith('.docx')) {
        try {
          const result = await mammoth.extractRawText({ arrayBuffer: event.target.result as ArrayBuffer });
          resolve(result.value);
        } catch (error) {
          console.error('Error parsing .docx file:', error);
          reject(new Error('Could not parse the .docx file.'));
        }
      } else {
        resolve(event.target.result as string);
      }
    };

    reader.onerror = (error) => {
      console.error('FileReader error:', error);
      reject(new Error('Error reading file.'));
    };
    
    if (file.name.endsWith('.docx')) {
        reader.readAsArrayBuffer(file);
    } else {
        reader.readAsText(file);
    }
  });
}

export function fileToBase64(file: File): Promise<{ base64: string; mimeType: string }> {
  return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => {
          const result = reader.result as string;
          // result is a data URL like "data:image/jpeg;base64,..."
          // We need to strip the prefix for the API
          const base64 = result.split(',')[1];
          resolve({ base64, mimeType: file.type });
      };
      reader.onerror = error => reject(error);
  });
}


export async function downloadImagesAsZip(images: GeneratedImage[]): Promise<void> {
    const successfulImages = images.filter(img => img.base64);
    if (successfulImages.length === 0) return;

    const zip = new JSZip();
    
    successfulImages.forEach((image, index) => {
        if (image.base64) {
            // Use the engine name in the filename for clarity
            const filename = `image_${image.engine || 'unknown'}_${index + 1}.jpeg`;
            zip.file(filename, image.base64, { base64: true });
        }
    });

    try {
        const content = await zip.generateAsync({ type: 'blob' });
        saveAs(content, 'built_by_sadeem_images.zip');
    } catch(e) {
        console.error("Error creating zip file", e);
    }
}