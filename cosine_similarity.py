import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer

# Load the model for computing embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')

def compute_similarity(record, bot_name):
    # Dynamically access the target response and bot response columns based on the bot name
    answer_orig = str(record[f'{bot_name}_Target_Response'])
    answer_llm = str(record[f'{bot_name}_Response'])
    
    v_llm = model.encode(answer_llm)
    v_orig = model.encode(answer_orig)
    
    # Cosine similarity is the dot product of normalized vectors
    similarity = (v_llm @ v_orig) / (np.linalg.norm(v_llm) * np.linalg.norm(v_orig))
    return similarity

def process_bot(df, bot_name):
    dict_bot = df.to_dict(orient='records')
    similarity = []
    
    for record in tqdm(dict_bot, desc=f"Processing {bot_name}"):
        sim = compute_similarity(record, bot_name)
        similarity.append(sim)
    
    # Add the computed cosine similarity to the DataFrame
    df[f'cosine_{bot_name.lower()}'] = similarity
    return df

def save_bot_data(df, bot_name):
    # Select the columns relevant to the specific bot (including cosine similarity)
    bot_df = df[['Question', f'{bot_name}_Target_Response', f'{bot_name}_Response', f'cosine_{bot_name.lower()}']]
    
    # Save the bot's data to a separate CSV file
    bot_df.to_csv(f'{bot_name.lower()}_conversation_with_cosine.csv', index=False, encoding='utf-8')
    print(f"Saved {bot_name}'s data to '{bot_name.lower()}_conversation_with_cosine.csv'")

def main():
    df = load_data('data/bot_responses.csv')
    
    bot_names = ['Bella', 'Mia', 'Mike', 'Olivia']
    
    for bot in bot_names:
        # Compute similarity for each bot and update the dataframe
        df = process_bot(df, bot)
        
        # Save each bot's data separately
        save_bot_data(df, bot)
    
    # Calculate and print average cosine similarity for each bot
    for bot in bot_names:
        avg_similarity = df[f'cosine_{bot.lower()}'].mean()
        print(f"Average cosine similarity for {bot}: {avg_similarity:.4f}")

if __name__ == "__main__":
    main()
