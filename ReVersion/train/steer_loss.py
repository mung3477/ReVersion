import torch
import torch.nn.functional as F


def calculate_steer_loss(
    token_embedding,
	input_ids,
	placeholder_token_id,
	stop_ids,
	special_ids,
	positive_ids,
	temperature=0.07
):
	"""L_steer"""
	# compute input embeddings
	inputs_embeds = token_embedding(input_ids)  # (bs, 77, 768)
	positive_embeds = token_embedding(positive_ids)

	with torch.no_grad():  # no gradients from positive and negative embeds, only from <R>
		# compute entity embeds
		stop_mask = torch.isin(
			input_ids,
			torch.tensor(stop_ids + special_ids + [placeholder_token_id]).cuda()
		)  # (bs, 77)
		negative_embds = inputs_embeds[~stop_mask]  # (num_stop_tokens, 768)

		# remove bos and eos in positive embeddings
		pos_mask = torch.isin(positive_ids, torch.tensor(special_ids).cuda())  # (bs, 77)
		positive_embeds = positive_embeds[~pos_mask]  # (num_positive_tokens, 768), where num_positive_tokens = num_positives * bs

		# stack positives and negatives as a pn_block
		pn_embeds = torch.cat([positive_embeds, negative_embds], dim=0)
		pn_embeds_normalized = F.normalize(pn_embeds, p=2, dim=1)  # (num_positive_tokens+num_negative_tokens, 768)

	# compute relation embeds <R>
	relation_mask = (input_ids[0] == placeholder_token_id)  # (77)
	relation_embeds = inputs_embeds[0][relation_mask]  # (1, 768)
	relation_embeds_normalized = F.normalize(relation_embeds, p=2, dim=1)

	# compute Multi-Instance InfoNCE loss
	logits = torch.einsum('nc,mc->nm', [relation_embeds_normalized, pn_embeds_normalized])  # (1, num_positive_tokens+num_negative_tokens)

	logits /= temperature
	nominator = torch.logsumexp(logits[:, :positive_embeds.shape[0]], dim=1)
	denominator = torch.logsumexp(logits, dim=1)

	return torch.mean(denominator - nominator)
