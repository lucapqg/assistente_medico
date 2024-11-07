import { GoogleGenerativeAI } from "@google/generative-ai";
import { GoogleAICacheManager } from "@google/generative-ai/server";
import { pipeline } from '@xenova/transformers';
import {QdrantClient} from '@qdrant/qdrant-js';
import dotenv from 'dotenv';

dotenv.config();

const embedding_model = await pipeline('feature-extraction','Xenova/all-mpnet-base-v2');
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const cacheManager = new GoogleAICacheManager(process.env.GEMINI_API_KEY);
const client = new QdrantClient({
    url:process.env.QDRANT_API_URL,
    apiKey:process.env.QDRANT_API_KEY
});

// Query text
let query = "What is cancer?"


// Embedding
let features = await embedding_model(query);
let query_embedding = Array.from(features[0][0].data);

// Vector search
let vector_search = await client.search("cancer", {
    vector: query_embedding,
    limit: 15,
    with_payload:true,
    with_vectors:false,
});

// console.log(documents);
// Instruct text
const systemInstruction = "You are a Medical Assistant LLM specialized in providing medical information and guidance. Your primary goal is to offer accurate and contextually appropriate medical advice based on the information provided by the user. Always think step-by-step, ensuring your responses are clear, logical, and based on established medical knowledge. Do not provide information beyond the medical context of the inquiry."
const displayName = "MIRA";
const model = "models/gemini-1.5-flash-001";
let ttlSeconds = 5;
let documents = vector_search.map(docs => docs.payload["text"]);

const createCacheResult = await cacheManager.create({
    ttlSeconds,
    model,
    displayName,
    systemInstruction,
    contents: [{
        role: "user",
        parts: documents.map(doc => ({ text: doc }))
    }]
  });

const cacheServiceName = createCacheResult.name;
const totalTokenCount = createCacheResult.usageMetadata.totalTokenCount;
const expireTime = createCacheResult.expireTime;
const queriedCache = await cacheManager.get(cacheServiceName);

// Construct a `GenerativeModel` which uses the cache object.
const genModel = genAI.getGenerativeModelFromCachedContent(queriedCache);

// Run inference on the service.
const result = await genModel.generateContent({
contents: [
    {
    role: "user",
    parts: [
        { text: query },
    ],
    },
],
});

// The response should note that purple cats drink chicken soup, and comment on
// the picture of the cat as well.
console.log(result.response.text());
console.log(
    "created cache: ",
    cacheServiceName,
    " expires: ",
    expireTime,
    " total tokens: ",
    totalTokenCount,
  );
