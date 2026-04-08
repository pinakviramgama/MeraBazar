import { pipeline } from "@xenova/transformers";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

let extractor;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const generateEmbedding = async (imageBuffer) => {
  try {
    if (!extractor) {
      extractor = await pipeline(
        "image-feature-extraction",
        "Xenova/clip-vit-base-patch32"
      );
    }

    //  Create temp file
    const tempPath = path.join(__dirname, "temp-image.jpg");

    fs.writeFileSync(tempPath, imageBuffer);

    //  Pass file path instead of buffer
    const output = await extractor(tempPath);

    // Delete temp file after processing
    fs.unlinkSync(tempPath);

    const embedding = Array.from(output.data);

    return normalizeVector(embedding);

  } catch (error) {
    console.error("Embedding Error:", error);
    throw error;
  }
};

function normalizeVector(vector) {
  const magnitude = Math.sqrt(
    vector.reduce((sum, val) => sum + val * val, 0)
  );
  return vector.map((val) => val / magnitude);
}