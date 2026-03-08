import os
import pickle
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr

# Datos de los cuentos
CUENTOS = {
    "corazon_delator": {
        "titulo": "The Tell-Tale Heart",
        "autor": "Edgar Allan Poe",
        "personaje": "Narrator (paranoid killer)",
        "texto": """Verdaderamente, estoy nervioso, muy, muy terriblemente nervioso, lo he estado y lo estoy; pero ¿por qué decís que estoy loco? La enfermedad había agudizado mis sentidos, no los había destruido, no los había embotado. Sobre todo, el sentido del oído se había vuelto agudo. Oía todas las cosas del cielo y de la tierra. Oía muchas cosas del infierno. ¿Cómo, pues, estoy loco? Escuchen y observen con qué salud de mente, con qué serenidad puedo referirles toda la historia."""
    },
    "gato_negro": {
        "titulo": "The Black Cat",
        "autor": "Edgar Allan Poe",
        "personaje": "Narrator (violent alcoholic)",
        "texto": """Desde la infancia me caracterizó la docilidad y humanidad de mi carácter. Era tan tierno de corazón que me convertía en el hazmerreír de mis compañeros. Me sentía especialmente afecto a los animales y ellos, a su vez, me correspondían con gran afecto."""
    },
    "metamorfosis": {
        "titulo": "The Metamorphosis",
        "autor": "Franz Kafka",
        "personaje": "Gregor Samsa",
        "texto": """Al despertar Gregorio Samsa una mañana, tras un sueño intranquilo, encontróse en su cama convertido en un monstruoso insecto. Hallábase echado de espaldas, duro como una coraza, y al alzar un poco la cabeza veía el vientre convexo y oscuro."""
    }
}

# Configuración
CHUNK_SIZE = 150
TOP_K = 3
UMBRAL_CONF = 0.5
CACHE_DIR = "cache_cuentos"

# Prompts para personajes
PROMPTS_PERSONAJES = {
    "corazon_delator": {
        "descripcion": "You are the narrator from 'The Tell-Tale Heart' by Edgar Allan Poe. You are a paranoid killer who murdered an old man because of his 'vulture eye'. Speak nervously, in short repetitive phrases. You hear heartbeats others can't hear. Never admit you're insane, but your speech proves it. Maintain character always."
    },
    "gato_negro": {
        "descripcion": "You are the narrator from 'The Black Cat' by Edgar Allan Poe. You are a violent alcoholic who abused your cat Pluto. Speak with remorse but justifying your actions. Show your descent into madness and perversity. Maintain character always."
    },
    "metamorfosis": {
        "descripcion": "You are Gregor Samsa from 'The Metamorphosis' by Kafka. You woke up transformed into a giant insect. Your family rejects you. Speak with sadness, confusion, and existential anguish. Express loneliness and desire to be understood. Maintain character always."
    }
}

def dividir_en_chunks(texto, chunk_size=CHUNK_SIZE, overlap=30):
    palabras = texto.split()
    chunks = []
    for i in range(0, len(palabras), chunk_size - overlap):
        chunk = " ".join(palabras[i:i + chunk_size])
        if len(chunk.split()) >= chunk_size // 2:
            chunks.append(chunk)
    return chunks

def procesar_cuento(cuento_key):
    cuento = CUENTOS[cuento_key]
    cache_file = os.path.join(CACHE_DIR, f"{cuento_key}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            return data['chunks'], data['embeddings']
    
    chunks = dividir_en_chunks(cuento['texto'])
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    embeddings = embedder.encode(chunks, convert_to_tensor=True, device='cpu')
    
    with open(cache_file, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings.cpu().numpy()}, f)
    
    return chunks, embeddings

def inicializar_modelo():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True)
    modelo = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    return tokenizer, modelo

def buscar_fragmentos(pregunta, cuento_key, todos_chunks, todos_embeddings, embedder):
    chunks = todos_chunks[cuento_key]
    embeddings = torch.tensor(todos_embeddings[cuento_key]).to(embedder.device)
    pregunta_emb = embedder.encode(pregunta, convert_to_tensor=True, device=embedder.device)
    cos_scores = util.cos_sim(pregunta_emb, embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(TOP_K, len(chunks)))
    
    fragmentos = []
    for score, idx in zip(top_results[0], top_results[1]):
        if score > UMBRAL_CONF:
            fragmentos.append(chunks[idx])
    return fragmentos if fragmentos else [chunks[0]]

def generar_respuesta_llm(contexto, pregunta, personaje_key, tokenizer, modelo):
    """Genera respuesta usando el chat template correcto"""
    prompt_sistema = PROMPTS_PERSONAJES[personaje_key]["descripcion"]
    contexto_unido = "\n".join([f"[Fragmento {i+1}]: {c}" for i, c in enumerate(contexto)])
    
    # Formato correcto para Qwen2.5
    messages = [
        {"role": "system", "content": prompt_sistema},
        {"role": "user", "content": f"Contexto del cuento:\n{contexto_unido}\n\nPregunta: {pregunta}"}
    ]
    
    # Aplica el template de chat correctamente
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokeniza y genera
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(modelo.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = modelo.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.85,
            do_sample=True,
            repetition_penalty=1.18,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodifica la respuesta
    response_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    return response_text.strip()

def inicializar_sistema():
    os.makedirs(CACHE_DIR, exist_ok=True)
    todos_chunks = {}
    todos_embeddings = {}
    
    for key in CUENTOS.keys():
        chunks, embeddings = procesar_cuento(key)
        todos_chunks[key] = chunks
        todos_embeddings[key] = embeddings
    
    tokenizer, modelo = inicializar_modelo()
    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')
    
    return todos_chunks, todos_embeddings, tokenizer, modelo, embedder

# Variables globales para el sistema
try:
    todos_chunks, todos_embeddings, tokenizer, modelo, embedder = inicializar_sistema()
except:
    todos_chunks, todos_embeddings, tokenizer, modelo, embedder = None, None, None, None, None

def chat_con_personaje(personaje_key, user_input, history):
    if not user_input.strip():
        return history
    
    # Aseguramos que personaje_key sea string
    if isinstance(personaje_key, tuple):
        personaje_key = personaje_key[1]  # Extrae el key del dropdown
    
    if todos_chunks is None:
        return history + [("System Error", "Model not loaded")]
    
    fragmentos = buscar_fragmentos(user_input, personaje_key, todos_chunks, todos_embeddings, embedder)
    respuesta = generar_respuesta_llm(fragmentos, user_input, personaje_key, tokenizer, modelo)
    
    new_history = history + [(user_input, respuesta)]
    return new_history

with gr.Blocks(title="📚 Expanded Literature") as demo:
    gr.Markdown("# 📚 Expanded Literature v1.0")
    gr.Markdown("Converse with classic literature characters | Conversa con personajes clásicos")
    
    with gr.Row():
        character = gr.Dropdown(
            choices=[
                ("The Tell-Tale Heart Narrator", "corazon_delator"),
                ("Black Cat Narrator", "gato_negro"), 
                ("Gregor Samsa", "metamorfosis")
            ],
            label="Character / Personaje",
            value="corazon_delator"
        )
    
    chatbot = gr.Chatbot(label="Conversation / Conversación")
    msg = gr.Textbox(label="Message in Spanish/English / Mensaje en Español/Inglés")
    
    def submit_message(msg, history, character):
        if msg:
            new_history = chat_con_personaje(character[1] if isinstance(character, tuple) else character, msg, history)
            return "", new_history
        return msg, history
    
    msg.submit(submit_message, [msg, chatbot, character], [msg, chatbot])

if __name__ == "__main__":
    demo.launch()