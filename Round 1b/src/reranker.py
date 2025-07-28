# src/reranker.py
from sentence_transformers import util
import torch


def mmr_rerank(query_embedding, section_embeddings, sections, lambda_val=0.5, top_n=5):
    """
    Reranks a list of sections using Maximal Marginal Relevance (MMR).

    Args:
        query_embedding: The embedding of the search query.
        section_embeddings: A tensor of embeddings for all sections.
        sections: The list of section dictionaries.
        lambda_val: Balances relevance and diversity (0 to 1). 0.5 is a good start.
        top_n: The number of results to return.

    Returns:
        A list of the top_n reranked sections.
    """
    if not sections:
        return []

    # Calculate initial relevance scores
    relevance_scores = util.cos_sim(query_embedding, section_embeddings)[0]

    # Start with the most relevant section
    selected_indices = [torch.argmax(relevance_scores).item()]
    remaining_indices = [i for i in range(len(sections)) if i not in selected_indices]

    while len(selected_indices) < top_n and len(remaining_indices) > 0:
        mmr_scores = {}

        for index in remaining_indices:
            # Relevance score for this candidate
            relevance_to_query = relevance_scores[index]

            # Diversity score (find max similarity to already selected items)
            similarity_to_selected = util.cos_sim(
                section_embeddings[index], section_embeddings[selected_indices]
            )
            max_similarity = torch.max(similarity_to_selected).item()

            # Calculate MMR score
            mmr_score = (
                lambda_val * relevance_to_query - (1 - lambda_val) * max_similarity
            )
            mmr_scores[index] = mmr_score

        # Select the item with the highest MMR score
        best_index = max(mmr_scores, key=mmr_scores.get)
        selected_indices.append(best_index)
        remaining_indices.remove(best_index)

    # Return the sections corresponding to the selected indices
    reranked_sections = [sections[i] for i in selected_indices]

    # Assign the final importance rank
    for i, section in enumerate(reranked_sections):
        section["importance_rank"] = i + 1

    return reranked_sections
