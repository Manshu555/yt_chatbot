from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from youtube_transcript_api import YouTubeTranscriptApi
from chromadb import Client

# --- Global model/tokenizer ---
model = None
tokenizer = None
chroma_client = Client()

def load_model():
    global model, tokenizer
    model_name = "distilgpt2"  # Lightweight model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto"  # Automatically map to CPU/GPU if available
    )
    # Set pad_token_id if not already set (required for distilgpt2)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model loaded.")

def get_or_create_collection(video_id):
    collection_name = f"yt_transcript_{video_id}"

    try:
        collection = chroma_client.get_collection(collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        collection = chroma_client.create_collection(collection_name)
        print(f"Created new collection: {collection_name}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript_text = " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            print(f"Transcript fetch error: {e}")
            return None

        sentences = transcript_text.split('.')
        documents = [s.strip() for s in sentences if s.strip()]
        ids = [f"chunk_{i}" for i in range(len(documents))]

        if documents:
            print(f"Adding {len(documents)} chunks to ChromaDB for video {video_id}")
            collection.add(documents=documents, ids=ids)

    return collection

def query_transcript(query_text, video_id, n_results=5):
    global model, tokenizer

    print(f"\nQuery: {query_text} | Video ID: {video_id}")
    collection = get_or_create_collection(video_id)
    if not collection:
        return "Could not fetch transcript for the given video ID."

    results = collection.query(query_texts=[query_text], n_results=n_results)
    retrieved_chunks = results['documents'][0]

    if not retrieved_chunks:
        return "Could not find relevant information in the transcript."

    context = "\n".join(retrieved_chunks)
    prompt = f"""Using the following context, answer the question.
If the answer is not in the context, say "I could not find the answer in the transcript."
Context:
{context}
Question: {query_text}

Answer:
"""

    print("Generating response with LLM...")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_index = response.find("Answer:")
    return response[answer_start_index + len("Answer:"):].strip() if answer_start_index != -1 else response.strip()