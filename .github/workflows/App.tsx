import React, { useState, useCallback, useRef } from 'react';
import { Download, FileText, ImageIcon, ChevronsRight, AlertTriangle, X, Archive, Image as ImageIconLucide, Bookmark, UploadCloud, Trash2 } from 'lucide-react';
import { generatePrompts, generateImage, analyzeImageStyle } from './services/gemini';
import { generateImageWithStability } from './services/stability';
import { readTextFromFile, downloadImagesAsZip, fileToBase64 } from './utils/fileHelper';
import { GeneratedImage, AspectRatio, AppState } from './types';
import { Modal } from './components/Modal';

const ASPECT_RATIOS: AspectRatio[] = ["16:9", "9:16", "4:3", "3:4", "1:1"];

type PromptSource = 'script' | 'custom';

export default function App() {
  const [script, setScript] = useState('');
  const [niche, setNiche] = useState('');
  const [imageStyle, setImageStyle] = useState('cinematic, photorealistic, 4k');
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');
  const [generatedPrompts, setGeneratedPrompts] = useState<string[]>([]);
  const [generatedImages, setGeneratedImages] = useState<GeneratedImage[]>([]);
  const [appState, setAppState] = useState<AppState>(AppState.IDLE);
  const [progressMessage, setProgressMessage] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [imagePreviewModalOpen, setImagePreviewModalOpen] = useState(false);
  const [currentPreviewImage, setCurrentPreviewImage] = useState<GeneratedImage | null>(null);
  const [promptSource, setPromptSource] = useState<PromptSource>('script');
  const [customPrompts, setCustomPrompts] = useState('');

  const [referenceImageFile, setReferenceImageFile] = useState<File | null>(null);
  const [referenceImagePreviewUrl, setReferenceImagePreviewUrl] = useState<string | null>(null);

  const aiRequestPrompt = useRef('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const referenceImageInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      try {
        setScript(await readTextFromFile(file));
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to read file.');
        setScript('');
      }
    }
  };

  const handleReferenceImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
        setReferenceImageFile(file);
        setReferenceImagePreviewUrl(URL.createObjectURL(file));
        setError(null);
    } else if (file) {
        setError("Please upload a valid image file (e.g., JPG, PNG, WEBP).");
    }
  };

  const removeReferenceImage = () => {
    setReferenceImageFile(null);
    if(referenceImagePreviewUrl) {
        URL.revokeObjectURL(referenceImagePreviewUrl);
    }
    setReferenceImagePreviewUrl(null);
    if(referenceImageInputRef.current) {
        referenceImageInputRef.current.value = '';
    }
  };


  const handleGenerate = useCallback(async () => {
    setError(null);
    setProgressMessage('Starting generation...');
    setGeneratedImages([]);
    setGeneratedPrompts([]);
    aiRequestPrompt.current = '';

    try {
        let prompts: string[] = [];
        let effectiveImageStyle = imageStyle;

        setAppState(AppState.GENERATING_PROMPTS);

        // 1. Analyze reference image style if it exists to create a combined style guide.
        if (referenceImageFile) {
            setProgressMessage('Analyzing reference image style...');
            try {
                const imageData = await fileToBase64(referenceImageFile);
                const styleFromImage = await analyzeImageStyle(imageData);
                effectiveImageStyle = `${styleFromImage}, ${imageStyle}`;
            } catch (err) {
                const errorMessage = err instanceof Error ? err.message : 'Failed to analyze reference image.';
                setError(errorMessage);
                setAppState(AppState.IDLE);
                return;
            }
        }

        // 2. Get base prompts from script or custom input.
        if (promptSource === 'script') {
            if (!script.trim()) {
                setError('Script cannot be empty.');
                setAppState(AppState.IDLE);
                return;
            }
            setProgressMessage('Analyzing script and generating prompts...');
            // The combined style is passed to the AI to be baked into the generated prompts.
            const { prompts: generated, requestPrompt } = await generatePrompts(script, effectiveImageStyle, niche);
            prompts = generated;
            setGeneratedPrompts(prompts);
            aiRequestPrompt.current = requestPrompt;
        } else { // 'custom'
            if (!customPrompts.trim()) {
                setError('Custom prompts field cannot be empty.');
                setAppState(AppState.IDLE);
                return;
            }
            
            // Parse numbered prompts
            const promptRegex = /^\s*(\d+)\s*[:=\-.]\s*(.+)$/;
            const parsedPrompts: { index: number; text: string }[] = [];

            customPrompts.trim().split('\n').forEach(line => {
                const match = line.trim().match(promptRegex);
                if (match) {
                    parsedPrompts.push({
                        index: parseInt(match[1], 10),
                        text: match[2].trim()
                    });
                }
            });

            if (parsedPrompts.length === 0) {
                setError('No valid numbered prompts found. Use a format like "1. Your prompt".');
                setAppState(AppState.IDLE);
                return;
            }

            // Sort by index and extract prompt text
            prompts = parsedPrompts
                .sort((a, b) => a.index - b.index)
                .map(p => p.text);
            
            setGeneratedPrompts(prompts); // Show original prompts
        }
        
        if (prompts.length === 0) {
            setError("No prompts were generated or provided.");
            setAppState(AppState.IDLE);
            return;
        }
      
        // 3. Generate images from the prompts.
        setAppState(AppState.GENERATING_IMAGES);
        const imageResults: GeneratedImage[] = [];

        for (let i = 0; i < prompts.length; i++) {
            setProgressMessage(`Generating image ${i + 1} of ${prompts.length}...`);

            let finalPrompt = prompts[i];
            // For custom prompts, we append the style here. For script prompts, the style is already baked in.
            if (promptSource === 'custom' && effectiveImageStyle.trim()) {
                finalPrompt = `${prompts[i]}, ${effectiveImageStyle}`;
            }

            // First attempt with Google Gemini
            let result = await generateImage(finalPrompt, aspectRatio);
            let engine: GeneratedImage['engine'] = 'google';

            // Fallback to Stability AI if Google blocks the prompt for safety reasons
            if (result.error && result.error.includes('safety reasons') && process.env.STABILITY_API_KEY) {
                setProgressMessage(`Prompt ${i + 1} rejected by Google. Retrying with Stability AI...`);
                engine = 'stability'; // We are now attempting with stability
                result = await generateImageWithStability(finalPrompt, aspectRatio);
                
                // If fallback also failed, we enrich the error message.
                if (result.error) {
                    result.error = `Google blocked prompt. Stability AI also failed: ${result.error}`;
                }
            }


            let newImage: GeneratedImage;
            if (result.base64) {
                 newImage = {
                    prompt: prompts[i], // Show the original prompt
                    base64: result.base64,
                    url: `data:image/jpeg;base64,${result.base64}`,
                    engine,
                };
            } else {
                 newImage = {
                    prompt: prompts[i],
                    error: result.error || 'Failed to generate image.',
                    engine,
                };
            }
            
            imageResults.push(newImage);
            setGeneratedImages([...imageResults]); // Update state after each attempt
        }
      
        setAppState(AppState.DONE);
        const successfulImages = imageResults.filter(img => !img.error).length;
        if (successfulImages === prompts.length) {
            setProgressMessage('All images generated successfully!');
        } else {
            setProgressMessage(`Generated ${successfulImages} of ${prompts.length} images. Some prompts may have failed.`);
        }

    } catch (err) {
        console.error(err);
        const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
        setError(errorMessage);
        setAppState(AppState.IDLE);
    }
  }, [script, imageStyle, niche, aspectRatio, promptSource, customPrompts, referenceImageFile]);

  const downloadPrompts = () => {
    const blob = new Blob([generatedPrompts.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'generated_prompts.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openImagePreview = (image: GeneratedImage) => {
    setCurrentPreviewImage(image);
    setImagePreviewModalOpen(true);
  };
  
  const isLoading = appState === AppState.GENERATING_PROMPTS || appState === AppState.GENERATING_IMAGES;
  const isScriptMode = promptSource === 'script';

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">
            Built by Sadeem
          </h1>
          <p className="mt-2 text-lg text-gray-400">Your AI-powered script-to-image studio</p>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column: Inputs & Controls */}
          <div className="flex flex-col gap-6 p-6 bg-gray-800/50 rounded-2xl border border-gray-700 shadow-lg">
            
            <section>
              <label className="flex items-center text-lg font-semibold mb-2 text-gray-300">
                <FileText className="w-5 h-5 mr-2 text-primary-400" />
                1. Provide Content
              </label>
              <div className="flex bg-gray-900 border border-gray-700 rounded-lg p-1 mb-3">
                <button onClick={() => setPromptSource('script')} disabled={isLoading} className={`flex-1 p-2 text-sm font-semibold rounded-md transition-colors ${!isScriptMode ? 'text-gray-400 hover:bg-gray-700' : 'bg-primary-600 text-white'}`}>From Script</button>
                <button onClick={() => setPromptSource('custom')} disabled={isLoading} className={`flex-1 p-2 text-sm font-semibold rounded-md transition-colors ${isScriptMode ? 'text-gray-400 hover:bg-gray-700' : 'bg-primary-600 text-white'}`}>From Prompts</button>
              </div>

              {isScriptMode ? (
                <div className="flex flex-col gap-3">
                  <textarea
                    id="script-input"
                    value={script}
                    onChange={(e) => setScript(e.target.value)}
                    placeholder="Paste your script here, or upload a file. The AI will create a prompt for each paragraph or logical scene."
                    className="w-full h-40 p-3 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                    disabled={isLoading}
                  />
                  <input type="file" ref={fileInputRef} onChange={handleFileChange} accept=".txt,.docx" className="hidden" />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full text-center px-4 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 font-semibold rounded-lg transition-colors disabled:opacity-50"
                    disabled={isLoading}
                  >
                    Upload Script File (.txt, .docx)
                  </button>
                   <div>
                      <label htmlFor="niche-input" className="flex items-center text-sm font-semibold mb-2 text-gray-300">
                        <Bookmark className="w-4 h-4 mr-2 text-primary-400" />
                        Niche / Topic (Optional)
                      </label>
                      <input
                        id="niche-input"
                        type="text"
                        value={niche}
                        onChange={(e) => setNiche(e.target.value)}
                        placeholder="e.g., futuristic gadgets, ancient history"
                        className="w-full p-3 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                        disabled={isLoading || !isScriptMode}
                      />
                       <p className="text-xs text-gray-500 mt-1">Provide a topic to give the AI more context about the script.</p>
                    </div>
                </div>
              ) : (
                <textarea
                  id="custom-prompts-input"
                  value={customPrompts}
                  onChange={(e) => setCustomPrompts(e.target.value)}
                  placeholder="Use numbers to order your prompts, e.g., '1. A futuristic city' or '2=A robot'."
                  className="w-full h-60 p-3 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                  disabled={isLoading}
                />
              )}
            </section>
            
            <section>
              <h3 className="flex items-center text-lg font-semibold mb-2 text-gray-300">
                  <ImageIcon className="w-5 h-5 mr-2 text-primary-400" />
                  2. Define Image Style
              </h3>
              <div className="flex flex-col gap-4 p-4 bg-gray-900/50 border border-gray-700 rounded-lg">
                  <div>
                      <label className="text-sm font-semibold text-gray-300 mb-1 block">Reference Image (Optional)</label>
                      <p className="text-xs text-gray-500 mb-2">Upload an image to automatically analyze and apply its style.</p>
                      <input type="file" ref={referenceImageInputRef} onChange={handleReferenceImageChange} accept="image/*" className="hidden" />
                      {referenceImagePreviewUrl ? (
                          <div className="relative group w-full p-2 border border-dashed border-gray-600 rounded-lg flex items-center gap-3">
                              <img src={referenceImagePreviewUrl} alt="Reference Preview" className="w-16 h-16 object-cover rounded-md" />
                              <div className="flex-1 text-sm">
                                  <p className="font-semibold text-gray-200 truncate">{referenceImageFile?.name}</p>
                                  <p className="text-gray-400">{(referenceImageFile!.size / 1024).toFixed(1)} KB</p>
                              </div>
                              <button onClick={removeReferenceImage} disabled={isLoading} className="absolute top-1 right-1 p-1.5 bg-gray-800/50 hover:bg-red-900/70 rounded-full text-gray-400 hover:text-white transition-colors">
                                  <Trash2 className="w-4 h-4" />
                              </button>
                          </div>
                      ) : (
                          <button onClick={() => referenceImageInputRef.current?.click()} disabled={isLoading} className="w-full p-4 border-2 border-dashed border-gray-600 hover:border-primary-500 rounded-lg flex flex-col items-center justify-center text-gray-400 hover:text-primary-400 transition-colors disabled:opacity-50">
                              <UploadCloud className="w-8 h-8 mb-2" />
                              <span className="text-sm font-semibold">Upload Image</span>
                          </button>
                      )}
                  </div>
                  
                  <div className="h-px bg-gray-700"></div>

                  <div>
                      <label htmlFor="style-input" className="text-sm font-semibold text-gray-300 mb-1 block">Style Keywords</label>
                      <p className="text-xs text-gray-500 mb-2">Add keywords to combine with the reference style.</p>
                      <input
                          id="style-input"
                          type="text"
                          value={imageStyle}
                          onChange={(e) => setImageStyle(e.target.value)}
                          placeholder="e.g., highly detailed, cinematic, 4k"
                          className="w-full p-3 bg-gray-900 border border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-colors"
                          disabled={isLoading}
                      />
                  </div>
              </div>
            </section>

            <section>
               <h3 className="flex items-center text-lg font-semibold mb-3 text-gray-300">
                <ImageIconLucide className="w-5 h-5 mr-2 text-primary-400" />
                3. Aspect Ratio
              </h3>
              <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                {ASPECT_RATIOS.map((ratio) => (
                  <button
                    key={ratio}
                    onClick={() => setAspectRatio(ratio)}
                    className={`p-2 text-center rounded-lg font-medium transition-all ${
                      aspectRatio === ratio
                        ? 'bg-primary-600 text-white ring-2 ring-primary-400'
                        : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
                    }`}
                    disabled={isLoading}
                  >
                    {ratio}
                  </button>
                ))}
              </div>
            </section>
            
            <button
                onClick={handleGenerate}
                disabled={isLoading || (promptSource === 'script' && !script.trim()) || (promptSource === 'custom' && !customPrompts.trim())}
                className="w-full flex items-center justify-center gap-3 py-3 px-6 text-lg font-bold text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-transform transform active:scale-95 disabled:bg-gray-600 disabled:cursor-not-allowed disabled:scale-100 mt-4"
              >
                {isLoading ? (
                  <>
                    <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    Generate Images <ChevronsRight className="w-6 h-6" />
                  </>
                )}
            </button>
            {error && <div className="mt-2 p-3 bg-red-900/50 border border-red-700 text-red-300 rounded-lg">{error}</div>}
          </div>

          {/* Right Column: Outputs */}
          <div className="flex flex-col gap-6 p-6 bg-gray-800/50 rounded-2xl border border-gray-700 shadow-lg min-h-[500px]">
            {appState === AppState.IDLE && (
              <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
                <ImageIcon size={64} className="mb-4" />
                <h2 className="text-2xl font-semibold">Your generated content will appear here</h2>
                <p className="mt-2 max-w-sm">Fill in the details on the left and click "Generate Images" to start the magic.</p>
              </div>
            )}
            
            {(isLoading || appState === AppState.DONE) && (
              <>
                {isScriptMode && aiRequestPrompt.current && (
                    <div className="flex-shrink-0">
                        <h2 className="text-xl font-semibold text-gray-300 mb-2">AI Request for Prompts</h2>
                        <div className="max-h-40 p-3 bg-gray-900 border border-gray-600 rounded-lg overflow-y-auto">
                            <pre className="text-sm text-gray-400 whitespace-pre-wrap font-mono">
                                {aiRequestPrompt.current}
                            </pre>
                        </div>
                    </div>
                )}

                 {(generatedPrompts.length > 0 || isLoading) && (
                    <div className="flex-shrink-0">
                      <div className="flex justify-between items-center mb-2">
                        <h2 className="text-xl font-semibold text-gray-300">Generated Prompts ({generatedPrompts.length})</h2>
                        <div className="flex gap-2">
                            <button
                                onClick={downloadPrompts}
                                disabled={generatedPrompts.length === 0}
                                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-primary-700 hover:bg-primary-600 rounded-md transition-colors disabled:opacity-50"
                            >
                                <Download className="w-4 h-4" /> .txt
                            </button>
                        </div>
                      </div>
                      <div className="h-32 p-3 bg-gray-900 border border-gray-600 rounded-lg overflow-y-auto">
                        {appState === AppState.GENERATING_PROMPTS && <p className="text-gray-400">Generating...</p>}
                        {generatedPrompts.map((p, i) => <p key={i} className="text-sm text-gray-300 py-1 border-b border-gray-700/50 last:border-b-0">{p}</p>)}
                      </div>
                    </div>
                )}
                
                <div className="flex-grow flex flex-col">
                   <div className="flex justify-between items-center mb-2">
                        <h2 className="text-xl font-semibold text-gray-300">Generated Images</h2>
                        <button
                          onClick={() => downloadImagesAsZip(generatedImages)}
                          disabled={!generatedImages.some(img => !img.error) || isLoading}
                          className="flex items-center gap-2 px-3 py-1.5 text-sm bg-green-700 hover:bg-green-600 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                            <Archive className="w-4 h-4" /> Download All (.zip)
                        </button>
                   </div>
                   
                   {progressMessage && appState !== AppState.DONE && <div className="text-center py-2 text-primary-300">{progressMessage}</div>}
                   {progressMessage && appState === AppState.DONE && <div className="text-center py-2 text-green-400">{progressMessage}</div>}


                    <div className="flex-grow bg-gray-900 border border-gray-600 rounded-lg p-4 overflow-y-auto">
                        {generatedImages.length > 0 ? (
                           <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                {generatedImages.map((image, index) => (
                                    <div key={index} className="group relative aspect-w-1 aspect-h-1 bg-gray-800 rounded-lg overflow-hidden">
                                        {image.url ? (
                                            <>
                                                <img src={image.url} alt={image.prompt} className="w-full h-full object-cover"/>
                                                {image.engine && (
                                                    <span className={`absolute top-2 left-2 px-2 py-0.5 text-xs font-bold text-white rounded-full capitalize ${image.engine === 'google' ? 'bg-blue-600' : 'bg-purple-600'}`}>
                                                        {image.engine}
                                                    </span>
                                                )}
                                                <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity flex flex-col justify-end p-2 cursor-pointer" onClick={() => openImagePreview(image)}>
                                                    <p className="text-xs text-white truncate">{image.prompt}</p>
                                                     <a
                                                        href={image.url}
                                                        download={`image_${image.engine || 'unknown'}_${index + 1}.jpeg`}
                                                        onClick={(e) => e.stopPropagation()}
                                                        className="absolute top-2 right-2 p-1.5 bg-white/20 hover:bg-white/40 rounded-full text-white transition-colors"
                                                        title="Download Image"
                                                    >
                                                        <Download className="w-4 h-4"/>
                                                    </a>
                                                </div>
                                            </>
                                        ) : (
                                            <div className="w-full h-full flex flex-col items-center justify-center p-3 text-center bg-red-900/20 border border-red-800 rounded-lg">
                                                <AlertTriangle className="w-8 h-8 text-red-400 mb-2"/>
                                                <p className="text-sm font-semibold text-red-300">Generation Failed</p>
                                                {image.engine && <p className="text-xs text-gray-400 capitalize">via {image.engine}</p>}
                                                <p className="text-xs text-gray-400 mt-1">{image.error}</p>
                                                <p className="text-xs text-gray-500 mt-2 line-clamp-2">Prompt: {image.prompt}</p>
                                            </div>
                                        )}
                                    </div>
                                ))}
                           </div>
                        ) : (
                            <div className="flex items-center justify-center h-full text-gray-600">
                                {isLoading ? 'Starting image generation...' : 'Images will appear here.'}
                            </div>
                        )}
                   </div>
                </div>
              </>
            )}
          </div>
        </main>
      </div>

      {imagePreviewModalOpen && currentPreviewImage && (
         <Modal title="Image Preview" onClose={() => setImagePreviewModalOpen(false)} maxWidth="max-w-4xl">
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-shrink-0 md:w-2/3">
                <img src={currentPreviewImage.url!} alt={currentPreviewImage.prompt} className="w-full h-auto object-contain rounded-lg max-h-[70vh]"/>
              </div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-primary-400 mb-2">Prompt</h3>
                <p className="p-3 bg-gray-800 rounded-md text-gray-300 text-sm">{currentPreviewImage.prompt}</p>
                <a
                  href={currentPreviewImage.url!}
                  download={`image_${currentPreviewImage.engine || 'unknown'}_${generatedImages.filter(i => !i.error).findIndex(img => img.url === currentPreviewImage.url) + 1}.jpeg`}
                  className="mt-4 w-full flex items-center justify-center gap-2 py-2 px-4 text-md font-semibold text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors"
                >
                  <Download className="w-5 h-5"/> Download Image
                </a>
              </div>
            </div>
         </Modal>
      )}

    </div>
  );
}